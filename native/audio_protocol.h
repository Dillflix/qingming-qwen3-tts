#pragma once

#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__linux__)
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

namespace qingming::audio_protocol {

// QAF1: magic[4], kind:u32, request_id:u64, payload_bytes:u32, reserved:u32.
// All integers and native float32 samples are little endian. Control payloads
// are UTF-8 JSON. There is exactly one DONE or ERROR frame per request.
enum class Kind : std::uint32_t { audio=1, done=2, error=3 };
constexpr std::uint32_t max_payload=1024*1024;

inline std::array<unsigned char,24> header(Kind kind, std::uint64_t id, std::uint32_t size) {
    if(size>max_payload) throw std::runtime_error("audio frame exceeds 1 MiB");
    std::array<unsigned char,24> out{{'Q','A','F','1'}};
    auto put=[&](unsigned offset, std::uint64_t value, unsigned bytes) {
        for(unsigned i=0;i<bytes;++i) out[offset+i]=static_cast<unsigned char>(value>>(8*i));
    };
    put(4,static_cast<std::uint32_t>(kind),4);
    put(8,id,8);
    put(16,size,4);
    return out;
}

class Writer {
public:
    explicit Writer(int fd):fd_(fd) {
        static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
        static_assert(std::endian::native==std::endian::little);
#if defined(__linux__)
        if(fd<3) throw std::runtime_error("--audio-fd must be a dedicated writable descriptor >= 3");
        const int flags=fcntl(fd,F_GETFL);
        if(flags<0 || (flags&O_ACCMODE)==O_RDONLY || fcntl(fd,F_SETFL,flags|O_NONBLOCK)<0)
            throw std::runtime_error("--audio-fd is not writable");
        // Convert a disconnected reader into a bounded request failure.
        std::signal(SIGPIPE,SIG_IGN);
#else
        throw std::runtime_error("jsonl-v1 audio descriptors require Linux");
#endif
    }

    void frame(Kind kind, std::uint64_t id, const void* payload, std::uint32_t size) {
        const auto bytes=header(kind,id,size);
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        write_all(bytes.data(),bytes.size(),deadline);
        write_all(payload,size,deadline);
    }

    void audio(std::uint64_t id, const float* samples, std::size_t count) {
        if(count>max_payload/sizeof(float)) throw std::runtime_error("audio chunk too large");
        if(count) frame(Kind::audio,id,samples,static_cast<std::uint32_t>(count*sizeof(float)));
    }

private:
    int fd_;
    void write_all(const void* data, std::size_t size, std::chrono::steady_clock::time_point deadline) {
#if defined(__linux__)
        auto* bytes=static_cast<const unsigned char*>(data);
        while(size) {
            const auto remaining=std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline-std::chrono::steady_clock::now()).count();
            if(remaining<=0) throw std::runtime_error("audio pipe write timed out");
            pollfd p{fd_,POLLOUT,0};
            const int ready=poll(&p,1,static_cast<int>(remaining));
            if(ready<0 && errno==EINTR) continue;
            if(ready<=0 || (p.revents&(POLLERR|POLLHUP|POLLNVAL)))
                throw std::runtime_error("audio pipe is unavailable or timed out");
            const ssize_t written=write(fd_,bytes,size);
            if(written<0 && (errno==EINTR || errno==EAGAIN)) continue;
            if(written<=0) throw std::runtime_error("audio pipe write failed");
            bytes+=written; size-=static_cast<std::size_t>(written);
        }
#else
        (void)data; (void)size; (void)deadline;
        throw std::runtime_error("audio pipe requires Linux");
#endif
    }
};
} // namespace qingming::audio_protocol
