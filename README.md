# Rinha de Backend 2026 — Detecção de Fraude

Solução para a Rinha de Backend 2026. O desafio consiste em classificar transações financeiras como fraude ou não-fraude.

## Arquitetura geral

```
Cliente HTTP
     │
     ▼
C LB :9999  (round-robin, epoll, splice)
     │        via Unix Domain Socket
     ├──────────────────────────────────┐
     ▼                                  ▼
  api1 (Bun)                        api2 (Bun)
  /socks/api1.sock                  /socks/api2.sock
     │
     └── TCP raw parser → Vectorizer → IVF-Flat scorer → resposta pré-computada
```

Dois processos Bun independentes recebem requisições via **Unix Domain Sockets (UDS)**. O load balancer é um binário C estático (~170 LoC) com `epoll` edge-triggered que faz round-robin puro sem inspecionar o payload — sem banco de dados, cache externo ou qualquer dependência de rede entre as instâncias.

## Orçamento de recursos (docker-compose)

| Serviço  | CPU   | Memória |
|----------|-------|---------|
| api1     | 0.45  | 165 MB  |
| api2     | 0.45  | 165 MB  |
| lb       | 0.10  | 20 MB   |
| **Total**| **1.0**| **350 MB** |

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

**Inferência em dois estágios (single centroid search):**

Uma única chamada `findTopCentroids(FULL_NPROBE = 32)` vasculha todos os 2000 centroids com um **max-heap de tamanho 32** — O(NLIST × log NPROBE) vs. O(NLIST × NPROBE) de insertion sort. Os resultados ficam ordenados por distância e reutilizados nos dois passes abaixo.

1. **Fast path** (`FAST_NPROBE = 5`): vasculha apenas os 5 clusters mais próximos com L2 exato sobre `Int16Array`. Se o voto é unânime (0 ou 5 fraudes entre os top-5 vizinhos), retorna imediatamente — cobre ~97% das requisições. **Bbox pruning**: antes de varrer um cluster, calcula o lower-bound de distância até sua bounding box; se já é pior que o heap atual, o cluster inteiro é pulado. **Early exit parcial**: dentro do scan, a distância é acumulada em blocos de 4 dimensões — se após 4, 8 ou 12 dimensões já supera o pior do heap, o candidato é descartado.

2. **Full path** (`FULL_NPROBE = 32`): ativado quando o voto não é unânime após o fast path (~3% das requisições). Vasculha os 27 clusters restantes do ranking já calculado, sem nenhuma busca de centroid adicional.

**Heap máximo e score final:**

Durante o scan, um heap máximo de K=5 slots (`_hDist`, `_hLabel`) mantém os vizinhos mais próximos encontrados até agora. A raiz do heap é sempre o **pior** dos top-5 (maior distância), permitindo comparação O(1): se o novo candidato é pior que o atual pior, é descartado imediatamente; senão, substitui a raiz e o heap se reordena. Ambos os arrays são pré-alocados e reutilizados entre chamadas.

Ao final, `fraudCount` é o número de vizinhos no heap com label = 1 (fraude):

| fraudCount | score | decisão |
|------------|-------|---------|
| 0 | 0.0 | aprovado (fast path) |
| 1 | 0.2 | aprovado (→ full path) |
| 2 | 0.4 | aprovado (→ full path) |
| 3 | 0.6 | negado (→ full path) |
| 4 | 0.8 | negado (→ full path) |
| 5 | 1.0 | negado (fast path) |

### Respostas pré-computadas

O servidor pré-aloca **256 buffers HTTP completos** (incluindo headers), um para cada bucket de score quantizado em `[0, 1]`. A resposta é escrita diretamente no socket sem nenhuma serialização em tempo de execução:

```ts
const body = `{"approved":${score < THRESHOLD},"fraud_score":${score.toFixed(4)}}`;
RESPONSES[b] = Buffer.from(`HTTP/1.1 200 OK\r\nContent-Type: ...`);
```

### Warmup

50 iterações de warmup são executadas na inicialização para aquecer o JIT do JavaScriptCore (JSC — engine do Bun) antes de o LB marcar o serviço como healthy via healthcheck no socket UDS.

## Build do modelo (`scripts/build-ivf-flat.ts`)

Script offline que constrói o arquivo `resources/ivf-flat.bin` a partir de `resources/references.bin`. 

1. **k-means++** sobre 100k amostras → 2000 centroids (coarse quantizer)
2. Atribuição de todos os N vetores ao centroid mais próximo
3. Reordenação por cluster + avaliação em 10k exemplos (acurácia, F1, FP, FN)
4. Escrita do arquivo binário com layout fixo

