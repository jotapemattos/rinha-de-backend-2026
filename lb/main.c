// TCP -> UDS round-robin load balancer. Listens on :9999, byte-splices each
// accepted connection to one of two backend UDS sockets. Per-connection RR,
// edge-triggered epoll, no HTTP parsing. See docs/adr/0004-* for rationale.

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define LISTEN_PORT   9999
#define BACKLOG       1024
#define MAX_EVENTS    256
#define BUF_SIZE      8192
#define BACKEND_COUNT 2

static const char *const BACKENDS[BACKEND_COUNT] = {
    "/socks/api1.sock",
    "/socks/api2.sock",
};

typedef struct pair pair_t;

typedef struct end {
    int fd;
    struct end *peer;
    pair_t *parent;
    size_t pending_len;
    char pending[BUF_SIZE];
} end_t;

struct pair {
    end_t client;
    end_t backend;
};

static int epfd;
static int listener_fd;
static int listener_marker;
static unsigned rr;

static int epoll_set_end(end_t *e, uint32_t events, int op) {
    struct epoll_event ev = {
        .events = events | EPOLLET | EPOLLRDHUP,
        .data.ptr = e,
    };
    return epoll_ctl(epfd, op, e->fd, &ev);
}

static int flush_pending(end_t *e) {
    while (e->pending_len > 0) {
        ssize_t n = write(e->fd, e->pending, e->pending_len);
        if (n > 0) {
            if ((size_t)n < e->pending_len)
                memmove(e->pending, e->pending + n, e->pending_len - n);
            e->pending_len -= (size_t)n;
        } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return 1;
        } else {
            return -1;
        }
    }
    return 0;
}

static int on_writable(end_t *e) {
    int rc = flush_pending(e);
    if (rc < 0) return -1;
    if (rc == 0) {
        if (epoll_set_end(e, EPOLLIN, EPOLL_CTL_MOD) < 0) return -1;
    }
    return 0;
}

static int on_readable(end_t *src) {
    end_t *dst = src->peer;
    char buf[BUF_SIZE];
    for (;;) {
        ssize_t n = read(src->fd, buf, sizeof(buf));
        if (n == 0) return -1;
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
            return -1;
        }
        // Drain any prior overflow before this chunk so bytes stay in order.
        if (flush_pending(dst) < 0) return -1;
        size_t off = 0;
        if (dst->pending_len == 0) {
            while (off < (size_t)n) {
                ssize_t w = write(dst->fd, buf + off, (size_t)n - off);
                if (w > 0) off += (size_t)w;
                else if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
                else return -1;
            }
        }
        size_t remaining = (size_t)n - off;
        if (remaining > 0) {
            if (dst->pending_len + remaining > BUF_SIZE) return -1;
            memcpy(dst->pending + dst->pending_len, buf + off, remaining);
            dst->pending_len += remaining;
            if (epoll_set_end(dst, EPOLLIN | EPOLLOUT, EPOLL_CTL_MOD) < 0) return -1;
        }
    }
}

static int connect_backend(const char *path) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    struct sockaddr_un addr = { .sun_family = AF_UNIX };
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0 && errno != EINPROGRESS) {
        close(fd);
        return -1;
    }
    return fd;
}

static void close_pair(pair_t *p, struct epoll_event *batch, int from, int to) {
    for (int i = from; i < to; i++) {
        if (batch[i].data.ptr == &p->client || batch[i].data.ptr == &p->backend) {
            batch[i].data.ptr = NULL;
        }
    }
    if (p->client.fd >= 0) close(p->client.fd);
    if (p->backend.fd >= 0) close(p->backend.fd);
    free(p);
}

static void handle_accept(void) {
    for (;;) {
        int cfd = accept4(listener_fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            return;
        }
        int one = 1;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        const char *path = BACKENDS[rr++ % BACKEND_COUNT];
        int bfd = connect_backend(path);
        if (bfd < 0) { close(cfd); continue; }
        pair_t *p = calloc(1, sizeof(*p));
        if (!p) { close(cfd); close(bfd); continue; }
        p->client.fd = cfd;
        p->client.peer = &p->backend;
        p->client.parent = p;
        p->backend.fd = bfd;
        p->backend.peer = &p->client;
        p->backend.parent = p;
        if (epoll_set_end(&p->client, EPOLLIN, EPOLL_CTL_ADD) < 0 ||
            epoll_set_end(&p->backend, EPOLLIN, EPOLL_CTL_ADD) < 0) {
            close(cfd); close(bfd); free(p);
        }
    }
}

static int start_listener(void) {
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = htonl(INADDR_ANY),
        .sin_port = htons(LISTEN_PORT),
    };
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { close(fd); return -1; }
    if (listen(fd, BACKLOG) < 0) { close(fd); return -1; }
    return fd;
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    struct rlimit rl = { 65536, 65536 };
    setrlimit(RLIMIT_NOFILE, &rl);

    listener_fd = start_listener();
    if (listener_fd < 0) { perror("listen"); return 1; }

    epfd = epoll_create1(EPOLL_CLOEXEC);
    if (epfd < 0) { perror("epoll_create1"); return 1; }

    struct epoll_event lev = {
        .events = EPOLLIN | EPOLLET,
        .data.ptr = &listener_marker,
    };
    if (epoll_ctl(epfd, EPOLL_CTL_ADD, listener_fd, &lev) < 0) {
        perror("epoll_ctl listener");
        return 1;
    }

    struct epoll_event evs[MAX_EVENTS];
    for (;;) {
        int n = epoll_wait(epfd, evs, MAX_EVENTS, -1);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("epoll_wait");
            return 1;
        }
        for (int i = 0; i < n; i++) {
            void *p = evs[i].data.ptr;
            if (p == NULL) continue;
            if (p == &listener_marker) { handle_accept(); continue; }
            end_t *e = (end_t *)p;
            uint32_t f = evs[i].events;
            int rc = 0;
            if (f & EPOLLERR) rc = -1;
            if (rc == 0 && (f & EPOLLOUT)) rc = on_writable(e);
            if (rc == 0 && (f & EPOLLIN)) rc = on_readable(e);
            if (rc == 0 && (f & (EPOLLHUP | EPOLLRDHUP))) rc = -1;
            if (rc < 0) close_pair(e->parent, evs, i + 1, n);
        }
    }
}
