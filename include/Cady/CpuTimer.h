#ifndef INCLUDE_CPU_TIMER_H
#define INCLUDE_CPU_TIMER_H

#include "Cady.h"

#include <chrono>
#include <string>

namespace Cady {

    struct cpu_timer {
        cpu_timer()
            : start_{ std::chrono::high_resolution_clock::now() }
        {}
        std::string format()const {
            double ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_).count());
            double seconds = ms / 1000.0;
            return std::to_string(seconds) + " seconds";
        }
        long long count()const
        {
            return (std::chrono::high_resolution_clock::now() - start_).count();
        }
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };

} // end namespace Cady

#endif // INCLUDE_CPU_TIMER_H