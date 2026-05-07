# Rinha de Backend 2026 — Detecção de Fraude

Solução para a Rinha de Backend 2026. O desafio consiste em classificar transações financeiras como fraude ou não-fraude com altíssima throughput e latência p99 < 1ms.

## Arquitetura geral

```
Cliente HTTP
     │
     ▼
HAProxy :9999  (round-robin, keep-alive)
     │           via Unix Domain Socket
     ├──────────────────────────────────┐
     ▼                                  ▼
  api1 (Bun)                        api2 (Bun)
  /socks/api1.sock                  /socks/api2.sock
     │
     └── TCP raw parser → Vectorizer → IVF-Flat scorer → resposta pré-computada
```

Dois processos Bun independentes recebem requisições via **Unix Domain Sockets (UDS)**. O HAProxy faz o balanceamento em round-robin sem banco de dados, cache externo ou qualquer dependência de rede entre as instâncias.

## Orçamento de recursos (docker-compose)

| Serviço  | CPU   | Memória |
|----------|-------|---------|
| api1     | 0.45  | 160 MB  |
| api2     | 0.45  | 160 MB  |
| haproxy  | 0.10  | 20 MB   |
| **Total**| **1.0**| **340 MB** |

Os sockets ficam em um volume `tmpfs` compartilhado entre os containers, zerando a latência de I/O de rede.

## Pipeline de inferência

Cada requisição passa por três etapas síncronas no mesmo thread:

### 1. Parser HTTP raw (`src/server.ts`)

O servidor **não usa nenhum framework HTTP**. Implementa diretamente `Bun.listen` sobre o socket Unix e faz parsing manual do protocolo HTTP/1.1 no buffer de bytes:

- `findHeaderEnd` — localiza `\r\n\r\n` com busca byte a byte.
- `parseContentLength` — lê o header `Content-Length` com comparação de bytes (case-insensitive) sem criar strings intermediárias.
- Requisições `GET /ready` são detectadas pelo primeiro byte (`'G'` = 71) e respondem com um buffer estático pré-alocado.
- O estado de fragmentação de TCP (`pending`) é mantido por socket: como TCP é um protocolo de stream, uma requisição pode chegar em múltiplos chunks. O `pending` acumula os bytes recebidos até ter a mensagem HTTP completa.

### 2. Vetorização (`src/modules/fraud/vectorize.ts`)

Converte o JSON da transação em um vetor `Float32Array` de **14 dimensões** sem alocar objetos intermediários:

| Índice | Feature | Normalização |
|--------|---------|--------------|
| 0 | Valor da transação | `/ max_amount` |
| 1 | Número de parcelas | `/ max_installments` |
| 2 | Razão valor/média do cliente | `/ amount_vs_avg_ratio` |
| 3 | Hora UTC da transação | `/ 23` |
| 4 | Dia da semana (seg=0) | `/ 6` |
| 5 | Minutos desde última transação | `/ max_minutes` (-1 se nula) |
| 6 | Km da última transação até atual | `/ max_km` (-1 se nula) |
| 7 | Km do terminal até residência | `/ max_km` |
| 8 | Número de transações nas 24h | `/ max_tx_count_24h` |
| 9 | Terminal online | 0/1 |
| 10 | Cartão presente | 0/1 |
| 11 | Merchant desconhecido | 0/1 |
| 12 | Risco do MCC | lookup em `mcc_risk.json` |
| 13 | Valor médio do merchant | `/ max_merchant_avg_amount` |

`vectorizeFromText` parseia o JSON diretamente sobre a string sem `JSON.parse`, usando busca de chaves e leitura numérica posicional. 

`parseISOtoMs` converte timestamps ISO 8601 em milissegundos com aritmética pura de caracteres — sem `new Date()`.

### 3. IVF-Flat scorer (`src/modules/fraud/ivf-flat.ts`)

Classificador kNN aproximado (k=5) sobre o arquivo de referências pré-indexado.

**O que é IVF-Flat:** o espaço de 14 dimensões é particionado em `NLIST = 2000` clusters via k-means. Cada cluster tem um **centroid** (ponto central, média dos seus vetores). Na busca, a transação é comparada com os centroids para encontrar as regiões mais próximas, e só os vetores dentro dessas regiões são vasculhados — em vez de comparar com todos os N vetores de referência.

**Índice** (`resources/ivf-flat.bin`):
- 2000 centroids treinados por k-means++ sobre amostra de 100k vetores
- Vetores de referência armazenados como `Int16Array` (quantização escalar ×32767) organizados por cluster

**Inferência em dois estágios:**

1. **Fast path** (`FAST_NPROBE = 5`): encontra os 5 centroids mais próximos com insertion sort e vasculha esses clusters com L2 exato. Early termination: se a distância parcial (após 4 ou 8 dimensões) já supera o pior dos top-5 atuais, o candidato é descartado sem calcular o restante.

2. **Full path** (`FULL_NPROBE = 20`): ativado quando `fraudCount ∈ {2, 3}` após o fast path. Nesses casos o score seria 0.4 ou 0.6 — exatamente na fronteira do threshold — então 15 clusters adicionais são vasculhados para um resultado mais confiante.

**Heap máximo e score final:**

Durante o scan, um heap máximo de K=5 slots (`_hDist`, `_hLabel`) mantém os vizinhos mais próximos encontrados até agora. A raiz do heap é sempre o **pior** dos top-5 (maior distância), permitindo comparação O(1): se o novo candidato é pior que o atual pior, é descartado imediatamente; senão, substitui a raiz e o heap se reordena. Ambos os arrays são pré-alocados e reutilizados entre chamadas.

Ao final, `fraudCount` é o número de vizinhos no heap com label = 1 (fraude):

| fraudCount | score | decisão |
|------------|-------|---------|
| 0 | 0.0 | aprovado |
| 1 | 0.2 | aprovado |
| 2 | 0.4 | aprovado (borderline → full path) |
| 3 | 0.6 | negado (borderline → full path) |
| 4 | 0.8 | negado |
| 5 | 1.0 | negado |

### Respostas pré-computadas

O servidor pré-aloca **256 buffers HTTP completos** (incluindo headers), um para cada bucket de score quantizado em `[0, 1]`. A resposta é escrita diretamente no socket sem nenhuma serialização em tempo de execução:

```ts
const body = `{"approved":${score < THRESHOLD},"fraud_score":${score.toFixed(4)}}`;
RESPONSES[b] = Buffer.from(`HTTP/1.1 200 OK\r\nContent-Type: ...`);
```

### Warmup

50 iterações de warmup são executadas na inicialização para aquecer o JIT do JavaScriptCore (engine do Bun) antes de o HAProxy marcar o serviço como healthy.

## Build do modelo (`scripts/build-ivf-flat.ts`)

Script offline que constrói o arquivo `resources/ivf-flat.bin` a partir de `resources/references.bin`. Precisa ser executado uma vez antes de subir o servidor.

1. **k-means++** sobre 100k amostras → 2000 centroids (coarse quantizer)
2. Atribuição de todos os N vetores ao centroid mais próximo
3. **PQ training** (M=7 sub-quantizadores, K=256 codewords cada) sobre resíduos — usado apenas internamente para candidatos na etapa de avaliação; o runtime de inferência não usa PQ
4. Reordenação por cluster + avaliação em 10k exemplos (acurácia, F1, FP, FN)
5. Escrita do arquivo binário com layout fixo documentado no código

**Layout do arquivo:**
```
[u32 ×7: NLIST, N, NPROBE, M, K_PQ, KNN, DIMS]
[f32: coarse centroids — NLIST × DIMS]
[f32: PQ codebooks — M × K_PQ × SUB]   ← lido pelo build; ignorado pelo runtime
[u32: cluster sizes — NLIST]
[u32: cluster offsets — NLIST]
[u8:  PQ codes — N × M]                ← lido pelo build; ignorado pelo runtime
[u8:  labels — N]
[u8:  padding]
[i16: vectors — N × DIMS]              ← usado pelo runtime (L2 exato)
```

## Regra de negócio

```
approved = fraud_score < 0.6
```

O threshold 0.6 é fixo conforme as regras do desafio. Ajustes de precisão/recall devem ser feitos no modelo (features, tamanho do índice, nprobe), não no threshold.

## Desenvolvimento local

```bash
# Instalar dependências
bun install

# Construir o índice IVF (necessário uma vez)
bun scripts/build-ivf-flat.ts

# Rodar localmente
bun src/server.ts

# Docker
docker compose up
```

O health check espera que o socket Unix exista antes de o HAProxy iniciar o tráfego.
