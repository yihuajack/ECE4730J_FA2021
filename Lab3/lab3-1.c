#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>

int main(int argc, char **argv) {
    int n = atoi(argv[1]);  // 0 <= n <= 3
    size_t x;
    // scanf("%d", &n);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(n, &set);
    sched_setaffinity(0, sizeof(cpu_set_t), &set);
    for (size_t i = 0; i <= 500000000; ++i) {
        x = (i + 1) * 2;
    }
    return 0;
}
