#pragma once
#include "config.h"
#include "optix.h"
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

class Exception : public std::runtime_error {
public:
    Exception(const char* msg) : std::runtime_error(msg) {}

    Exception(OptixResult res, const char* msg) : std::runtime_error(createMessage(res, msg).c_str()) {}

private:
    std::string createMessage(OptixResult res, const char* msg) {
        std::ostringstream out;
        out << optixGetErrorName(res) << ": " << msg;
        return out.str();
    }
};

[[noreturn]] inline void assertFailMsg(const std::string& msg, const char* file, unsigned int line) {
    std::stringstream ss;
    ss << msg << ": " << file << " (" << line << ')';
    throw Exception(ss.str().c_str());
}

static bool fileExists(const char* path) {
    std::ifstream str(path);
    return static_cast<bool>(str);
}

static bool fileExists(const std::string& path) {
    return fileExists(path.c_str());
}

static std::string sampleInputFilePath(const char* fileName) {
    // Allow for overrides.
    static const char* directories[] = {
        PROJECT_PTX_DIR,
        ".",
#if defined(_DEBUG) || defined(DEBUG)
        PROJECT_PTX_DIR "/Debug",
#else
        PROJECT_PTX_DIR "/Release",
#endif
    };

    std::vector<std::string> locations;
    for (const char* directory : directories) {
        if (directory) {
            std::string path = directory;
            path += '/';
            path += fileName;
            path.erase(path.size() - 3);
            path += ".optixir";
            locations.push_back(path);
            if (fileExists(path)) {
                return path;
            }
        }
    }

    std::string error = "sampleInputFilePath couldn't locate " + std::string(fileName) + " for sample "
                      + " in the following locations:\n";
    for (const auto& path : locations) {
        error += "\t" + path + "\n";
    }

    throw Exception(error.c_str());
}

static bool readSourceFile(std::string& str, const std::string& filename) {
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 1;
    } else {

        // Determine the file size
        file.seekg(0, std::ios::end);
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        // Resize the string to fit the file content
        str.resize(fileSize);

        // Read the file into the string
        if (!file.read(&str[0], fileSize)) {
            throw std::runtime_error("Error: Could not read file " + filename);
        }

        file.close();
        return true;
    }
    return false;
}

static void getInputDataFromFile(std::string& inputData, const char* filename) {
    const std::string sourceFilePath = sampleInputFilePath(filename);

    // Try to open source file
    if (!readSourceFile(inputData, sourceFilePath)) {
        std::string err = "Couldn't open source file " + sourceFilePath;
        throw std::runtime_error(err.c_str());
    }
}


struct SourceCache {
    std::map<std::string, std::string*> map;
    ~SourceCache() {
        for (std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it) {
            delete it->second;
        }
    }
};
static SourceCache g_sourceCache;

const char* getInputData(const char* filename, size_t& dataSize, const char** log = NULL) {
    if (log) {
        *log = NULL;
    }

    std::string *inputData, cu;
    std::string key                                    = std::string(filename) + ";";
    std::map<std::string, std::string*>::iterator elem = g_sourceCache.map.find(key);

    if (elem == g_sourceCache.map.end()) {
        inputData = new std::string();
        getInputDataFromFile(*inputData, filename);
        g_sourceCache.map[key] = inputData;
    } else {
        inputData = elem->second;
    }
    dataSize = inputData->size();
    return inputData->c_str();
}
