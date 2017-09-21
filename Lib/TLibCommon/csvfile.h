//
// Created by Pharrell_WANG on 20/2/2017.
//

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpragma-once-outside-header"
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"
#pragma ide diagnostic ignored "IncompatibleTypes"
#ifndef HTM_CSVFILE_H
#define HTM_CSVFILE_H


#pragma once

#include <string>
#include <iostream>
#include <fstream>

class csvfile;

inline static csvfile &endrow(csvfile &file);

inline static csvfile &flush(csvfile &file);

class csvfile {
    std::ofstream fs_;
    const std::string separator_;
public:
    csvfile(const std::string filename, const std::string separator = ",")
            : fs_(), separator_(separator) {
        fs_.exceptions(std::ios::failbit | std::ios::badbit);
        fs_.open(filename, std::ios::app);
    }

    ~csvfile() {
        flush();
        fs_.close();
    }

    void flush() {
        fs_.flush();
    }

    void endrow() {
        fs_ << std::endl;
    }

    csvfile &operator<<(csvfile &(*val)(csvfile &)) {
        return val(*this);
    }

//    csvfile &operator<<(const char *val) {
//        fs_ << '"' << val << '"' << separator_;
//        return *this;
//    }
//
//    csvfile &operator<<(const std::string &val) {
//        fs_ << '"' << val << '"' << separator_;
//        return *this;
//    }

    template<typename T>
    csvfile &operator<<(const T &val) {
        fs_ << val << separator_;
        return *this;
    }
};


inline static csvfile &endrow(csvfile &file) {
    file.endrow();
    return file;
}

inline static csvfile &flush(csvfile &file) {
    file.flush();
    return file;
}



#endif //HTM_CSVFILE_H

#pragma clang diagnostic pop