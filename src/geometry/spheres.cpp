// loadSpheres: sphere files, format "xyzr" (rules in rtcd/geometry.h). Numbers are parsed with
// std::istream >> float in the classic locale, as the old RTCD::meshModel::loadSphers did, so the float values are
// identical to the old stack's sphere arrays.

#include <rtcd/errors.h>
#include <rtcd/geometry.h>

#include <fstream>
#include <locale>
#include <sstream>
#include <string>
#include <system_error>

namespace rtcd {

    std::vector<Sphere> loadSpheres(const std::filesystem::path& file, std::string_view format) {
        if (format != "xyzr") {
            throw UnsupportedError("loadSpheres: sphere format '" + std::string(format) + "' is not supported (only \"xyzr\"), file "
                                   + file.string());
        }
        std::error_code ec;
        if (!std::filesystem::is_regular_file(file, ec)) {
            throw IoError("sphere file does not exist or is not a regular file", file);
        }
        std::ifstream in(file);
        if (!in) {
            throw IoError("cannot open sphere file", file);
        }

        std::vector<Sphere> spheres;
        std::string line;
        std::size_t lineNo = 0;
        while (std::getline(in, line)) {
            ++lineNo;
            if (lineNo == 1 && line.size() >= 3 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
                line.erase(0, 3); // UTF-8 byte order mark
            }
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.find_first_not_of(" \t") == std::string::npos || line[0] == '#') {
                continue; // empty (or blank) line, comment
            }
            std::istringstream iss(line);
            iss.imbue(std::locale::classic());
            float x = 0, y = 0, z = 0, r = 0;
            if (!(iss >> x >> y >> z >> r)) {
                throw IoError("expected \"x y z r\" (four numbers), got \"" + line + "\"", file, lineNo);
            }
            std::string rest;
            if (iss >> rest) {
                throw IoError("unexpected text \"" + rest + "\" after \"x y z r\"", file, lineNo);
            }
            if (!(r > 0.0f)) {
                throw IoError("sphere radius " + std::to_string(r) + " must be > 0", file, lineNo);
            }
            spheres.push_back(Sphere{Vec3{x, y, z}, r});
        }
        if (in.bad()) {
            throw IoError("read error", file, lineNo);
        }
        return spheres;
    }

} // namespace rtcd
