#pragma once

#include <array>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

// Offline Base profiles use exactly 2048 little-endian BF16 values. Model and
// reference provenance belongs in the profile manifest; the raw payload alone
// cannot establish that two checkpoints use the same speaker encoder.
namespace qingming::speaker_embedding {

inline constexpr std::size_t dimensions=2048;
inline constexpr std::size_t byte_count=dimensions*2;

inline void validate_path(const std::filesystem::path& path) {
    const auto& value=path.native();
    if(value.empty() || value.find(typename std::filesystem::path::value_type{})!=value.npos)
        throw std::runtime_error("speaker embedding path must be nonempty and contain no NUL");
}

inline void validate_sources(
    const std::filesystem::path& reference,
    const std::filesystem::path& input,
    const std::filesystem::path& output) {
    if(reference.empty()==input.empty())
        throw std::runtime_error("base-xvector requires exactly one of --ref-audio or --speaker-embedding-bf16");
    if(!reference.empty()) validate_path(reference);
    if(!input.empty()) validate_path(input);
    if(!output.empty()) {
        validate_path(output);
        if(reference.empty())
            throw std::runtime_error("--save-speaker-embedding-bf16 requires --ref-audio");
    }
}

inline void validate_audio_output(
    const std::filesystem::path& input,
    const std::filesystem::path& output,
    const std::filesystem::path& audio) {
    auto resolved=[](const std::filesystem::path& path) {
        const auto absolute=std::filesystem::absolute(path).lexically_normal();
        return std::filesystem::exists(absolute)
            ?std::filesystem::canonical(absolute)
            :std::filesystem::weakly_canonical(absolute.parent_path())/absolute.filename();
    };
    const auto wav=resolved(audio);
    for(const auto& embedding:{input,output})
        if(!embedding.empty() &&
           (resolved(embedding)==wav ||
            (std::filesystem::exists(embedding) && std::filesystem::exists(audio)
             &&std::filesystem::equivalent(embedding,audio))))
            throw std::runtime_error("speaker embedding path must differ from the output WAV path");
}

inline void validate(const std::vector<std::uint16_t>& values) {
    if(values.size()!=dimensions)
        throw std::runtime_error("speaker embedding must contain exactly 2048 BF16 values (4096 bytes)");
    bool nonzero=false;
    for(const auto value:values) {
        if((value&0x7f80u)==0x7f80u)
            throw std::runtime_error("speaker embedding contains a non-finite BF16 value");
        nonzero=nonzero || (value&0x7fffu)!=0;
    }
    if(!nonzero)
        throw std::runtime_error("speaker embedding must not be all zero");
}

inline std::vector<std::uint16_t> read(const std::filesystem::path& path) {
    validate_path(path);
    if(!std::filesystem::is_regular_file(path))
        throw std::runtime_error("speaker embedding is not a regular file: "+path.string());
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=static_cast<std::streamoff>(byte_count))
        throw std::runtime_error("speaker embedding must be exactly 4096 bytes: "+path.string());
    file.seekg(0);
    std::array<unsigned char,byte_count> bytes{};
    if(!file.read(reinterpret_cast<char*>(bytes.data()),bytes.size()) ||
       file.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("failed reading 4096-byte speaker embedding: "+path.string());
    std::vector<std::uint16_t> values(dimensions);
    for(std::size_t i=0;i<dimensions;++i)
        values[i]=static_cast<std::uint16_t>(bytes[2*i]|(static_cast<unsigned>(bytes[2*i+1])<<8));
    validate(values);
    return values;
}

// JSONL carries the registered payload, never a caller-controlled file path.
inline std::vector<std::uint16_t> from_hex(const std::string& hex) {
    if(hex.size()!=byte_count*2) throw std::runtime_error("invalid speaker embedding hex length");
    auto digit=[](char c)->unsigned {
        if(c>='0'&&c<='9') return c-'0';
        if(c>='a'&&c<='f') return c-'a'+10;
        throw std::runtime_error("invalid speaker embedding hex digit");
    };
    std::vector<std::uint16_t> values(dimensions);
    for(std::size_t i=0;i<dimensions;++i) {
        const auto low=(digit(hex[4*i])<<4)|digit(hex[4*i+1]);
        const auto high=(digit(hex[4*i+2])<<4)|digit(hex[4*i+3]);
        values[i]=static_cast<std::uint16_t>(low|(high<<8));
    }
    validate(values);
    return values;
}

inline void write_exclusive(
    const std::filesystem::path& path,
    const std::vector<std::uint16_t>& values) {
    validate_path(path);
    validate(values);
    std::array<unsigned char,byte_count> bytes{};
    for(std::size_t i=0;i<dimensions;++i) {
        bytes[2*i]=static_cast<unsigned char>(values[i]&0xffu);
        bytes[2*i+1]=static_cast<unsigned char>(values[i]>>8);
    }
#ifdef _WIN32
    int fd=_wopen(path.c_str(),_O_WRONLY|_O_CREAT|_O_EXCL|_O_BINARY,_S_IREAD|_S_IWRITE);
#else
    int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
#endif
    if(fd<0)
        throw std::runtime_error("cannot exclusively create speaker embedding (destination must not exist): "+path.string());
    try {
        std::size_t offset=0;
        while(offset<bytes.size()) {
#ifdef _WIN32
            const auto count=_write(fd,bytes.data()+offset,static_cast<unsigned>(bytes.size()-offset));
#else
            const auto count=::write(fd,bytes.data()+offset,bytes.size()-offset);
#endif
            if(count<0 && errno==EINTR) continue;
            if(count<=0) throw std::runtime_error("failed writing speaker embedding: "+path.string());
            offset+=static_cast<std::size_t>(count);
        }
#ifdef _WIN32
        const int synced=_commit(fd);
        const int closed=_close(fd);
#else
        const int synced=::fsync(fd);
        const int closed=::close(fd);
#endif
        fd=-1;
        if(synced!=0 || closed!=0)
            throw std::runtime_error("failed flushing speaker embedding: "+path.string());
    } catch(...) {
        if(fd>=0) {
#ifdef _WIN32
            _close(fd);
#else
            ::close(fd);
#endif
        }
        std::error_code ignored;
        std::filesystem::remove(path,ignored);
        throw;
    }
}

} // namespace qingming::speaker_embedding
