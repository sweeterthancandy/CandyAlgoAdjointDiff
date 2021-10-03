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
            return std::to_string(this->count());
        }
        long long count()const
        {
            return (std::chrono::high_resolution_clock::now() - start_).count();
        }
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };

} // end namespace Cady

#endif // INCLUDE_CPU_TIMER_H