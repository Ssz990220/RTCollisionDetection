// rtcd_core: Logger::defaultSink (rtcd/types.h).
#include <rtcd/types.h>

#include <cstdio>
#include <mutex>
#include <string>

namespace rtcd {

    namespace {
        const char* levelName(LogLevel l) {
            switch (l) {
            case LogLevel::Error:
                return "error";
            case LogLevel::Warning:
                return "warning";
            case LogLevel::Info:
                return "info";
            case LogLevel::Debug:
                return "debug";
            }
            return "?";
        }
    } // namespace

    // One line per message on stderr: "[rtcd][<level>] <message>". Serialized by a process-wide mutex so lines
    // from stream worker threads do not interleave. Never throws.
    void Logger::defaultSink(LogLevel l, std::string_view message) {
        static std::mutex m;
        try {
            std::string line = "[rtcd][";
            line += levelName(l);
            line += "] ";
            line.append(message.data(), message.size());
            if (line.empty() || line.back() != '\n') {
                line += '\n';
            }
            std::lock_guard<std::mutex> lock(m);
            std::fputs(line.c_str(), stderr);
            std::fflush(stderr);
        } catch (...) {
            // logging must never throw (it is called from destructors)
        }
    }

} // namespace rtcd
