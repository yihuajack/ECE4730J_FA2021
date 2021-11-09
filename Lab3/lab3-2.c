#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);  // 0 <= n <= 3
    if (n < 0 || n > 3) {
    	fprintf(stderr, "Error: Input CPU core number is invalid.\n");
    	// Return value of int main() other than EXIT_SUCCESS = 0 OR EXIT_FAILURE = 1
    	// in <stdlib.h> is not defined. Instructed by Lab_3.pdf, we return a negative value -1.
    	return -1;
    }
    // sched_get_priority_min(SCHED_RR) <= rtpriority <= sched_get_priority_max(SCHED_RR)
    int rtpriority = atoi(argv[2]);
    size_t x;
    if (rtpriority < sched_get_priority_min(SCHED_RR) || rtpriority > sched_get_priority_max(SCHED_RR)) {
    	fprintf(stderr, "Error: Input priority is out of range.\n");
    	return -1;
    }
    struct sched_param param;
    param.sched_priority = rtpriority;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(n, &set);
    sched_setaffinity(0, sizeof(cpu_set_t), &set);
    if (sched_setscheduler(0, SCHED_RR, &param)) {
        perror("Error: Fail to set scheduler SCHED_RR\n");
        return -1;
    }
    for (size_t i = 0; i <= 500000000; ++i) {
        x = (i + 1) * 2;
    }
    return 0;
}

