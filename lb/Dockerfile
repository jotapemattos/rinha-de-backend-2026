FROM alpine:3.20 AS builder
RUN apk add --no-cache gcc musl-dev
WORKDIR /src
COPY main.c .
RUN gcc -O2 -Wall -Wextra -static -o lb main.c && strip lb

FROM scratch
COPY --from=builder /src/lb /lb
EXPOSE 9999
ENTRYPOINT ["/lb"]
