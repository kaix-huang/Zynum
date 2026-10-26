// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#define _DARWIN_C_SOURCE 1
#include <signal.h>
#include <stdio.h>
#include <string.h>

#if !defined(__APPLE__) || !defined(__aarch64__)
#error "This lifecycle signal regression is verified only on native macOS AArch64"
#endif

extern int zynum_test_runtime_run(int persistent);
extern void zynum_test_runtime_shutdown(void);
extern int zynum_test_runtime_fault(int persistent, int allow_one);

static void sentinel_io(int signum) { (void)signum; }
static void sentinel_pipe(int signum) { (void)signum; }

static int same_action(const struct sigaction *a, const struct sigaction *b) {
    if (a->sa_handler != b->sa_handler || a->sa_flags != b->sa_flags) return 0;
    for (int i = 1; i < NSIG; ++i) {
        if (sigismember(&a->sa_mask, i) != sigismember(&b->sa_mask, i)) return 0;
    }
    return 1;
}

int zynum_test_runtime_main(int argc, char **argv) {
    if (argc != 2) return 2;
    const char *name = argv[1];
    int first_persistent = 0, second = 1, cycles = 1;
    if (!strcmp(name, "ordinary-persistent")) {}
    else if (!strcmp(name, "persistent-ordinary")) first_persistent = 1;
    else if (!strcmp(name, "ordinary-only")) second = 0;
    else if (!strcmp(name, "persistent-only")) { first_persistent = 1; second = 0; }
    else if (!strcmp(name, "restart-ordinary-persistent")) cycles = 2;
    else if (!strcmp(name, "restart-persistent-ordinary")) { first_persistent = 1; cycles = 2; }
    else return 2;

    struct sigaction original_io, original_pipe, expected_io, expected_pipe;
    struct sigaction io = {0}, pipe = {0};
    int status = 1, restores = 0;
    io.sa_handler = sentinel_io;
    pipe.sa_handler = sentinel_pipe;
    io.sa_flags = pipe.sa_flags = SA_RESTART;
    if (sigemptyset(&io.sa_mask) || sigemptyset(&pipe.sa_mask) ||
        sigaddset(&io.sa_mask, SIGUSR1) || sigaddset(&pipe.sa_mask, SIGUSR2)) return 3;
    if (sigaction(SIGIO, &io, &original_io)) return 3;
    if (sigaction(SIGPIPE, &pipe, &original_pipe)) {
        (void)sigaction(SIGIO, &original_io, NULL);
        return 3;
    }
    if (sigaction(SIGIO, NULL, &expected_io) || sigaction(SIGPIPE, NULL, &expected_pipe)) goto cleanup;

    for (int cycle = 0; cycle < cycles; ++cycle) {
        const int first = first_persistent ^ cycle;
        int work = zynum_test_runtime_run(first);
        if (!work && second) work = zynum_test_runtime_run(!first);
        zynum_test_runtime_shutdown();
        if (work) {
            fprintf(stderr, "lifecycle %s cycle %d: work/helper check failed (%d)\n", name, cycle + 1, work);
            goto cleanup;
        }
        if (sigaction(SIGIO, NULL, &io) || sigaction(SIGPIPE, NULL, &pipe)) goto cleanup;
        if (!same_action(&io, &expected_io) || !same_action(&pipe, &expected_pipe)) {
            fprintf(stderr, "lifecycle %s cycle %d: SIGIO restored=%d SIGPIPE restored=%d\n",
                name, cycle + 1, same_action(&io, &expected_io), same_action(&pipe, &expected_pipe));
            goto cleanup;
        }
        ++restores;
    }
    // Each local owner is fully reset before the next one initializes.
    for (int persistent = 0; persistent < 2; ++persistent) {
        for (int allow_one = 0; allow_one < 2; ++allow_one) {
            const int fault = zynum_test_runtime_fault(persistent, allow_one);
            if (fault) {
                fprintf(stderr, "lifecycle %s fault %d/%d failed (%d)\n", name, persistent, allow_one, fault);
                goto cleanup;
            }
            if (sigaction(SIGIO, NULL, &io) || sigaction(SIGPIPE, NULL, &pipe)) goto cleanup;
            if (!same_action(&io, &expected_io) || !same_action(&pipe, &expected_pipe)) {
                fprintf(stderr, "lifecycle %s fault %d/%d failed signal restoration\n", name, persistent, allow_one);
                goto cleanup;
            }
        }
    }
    status = 0;
cleanup:
    zynum_test_runtime_shutdown();
    if (sigaction(SIGIO, &original_io, NULL) || sigaction(SIGPIPE, &original_pipe, NULL)) status = 3;
    if (status == 0)
        printf("ZYNUM-RUNTIME-LIFECYCLE-V1 case=%s restores=%d status=ok\n", name, restores);
    return status;
}
