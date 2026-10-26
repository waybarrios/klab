// Native experiment launcher. CUDA is reached through native/driver.cpp.
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

bool is_project(const fs::path& path) {
    return fs::is_regular_file(path / "experiments.py") && fs::is_regular_file(path / "kernel_lab.py");
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && (std::string(argv[1]) == "version" || std::string(argv[1]) == "--version")) {
            std::cout << "klab 0.2.0\n";
            return 0;
        }
        int first = 1;
        fs::path root;
        if (argc > 1 && std::string(argv[1]) == "--project") {
            if (argc < 3) throw std::runtime_error("usage: klab --project <directory> <command>");
            root = argv[2];
            first = 3;
        } else if (const char* env = std::getenv("KERNEL_LAB_ROOT")) {
            root = env;
        } else {
            root = KLAB_PROJECT_ROOT;
        }
        root = fs::canonical(root);
        if (!is_project(root)) throw std::runtime_error(root.string() + " is not a kernel-lab project");
        const std::string python = (root / ".venv/bin/python").string();
        if (!fs::is_regular_file(python)) throw std::runtime_error("Missing " + python + "; follow README.md");
        std::string script = (root / "experiments.py").string();
        std::vector<char*> args{const_cast<char*>(python.c_str()), script.data()};
        for (int i = first; i < argc; ++i) args.push_back(argv[i]);
        args.push_back(nullptr);
        fs::current_path(root);
        execv(python.c_str(), args.data());
        throw std::runtime_error(std::string("Cannot start experiment runner: ") + std::strerror(errno));
    } catch (const std::exception& error) {
        std::cerr << "klab: " << error.what() << '\n';
        return 2;
    }
}
