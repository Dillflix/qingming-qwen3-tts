// Included inside qingming::production after the legacy result formatters.
// jsonl-v1 is deliberately CustomVoice-only; legacy CLI protocols are unchanged.
static int run_customvoice_jsonl(
    qwen3_tts::baseline::ResidentEngine& engine,
    qingming::audio_protocol::Writer& audio,
    std::size_t capacity,
    const std::string& partition) {
    using J=qwen3_tts::baseline::model_frontend::J;
    using Kind=qingming::audio_protocol::Kind;
    int device=0;
    hipDeviceProp_t properties{};
    if(hipGetDevice(&device)!=hipSuccess || hipGetDeviceProperties(&properties,device)!=hipSuccess)
        throw std::runtime_error("cannot identify resident HIP device");
    std::cout<<"{\"event\":\"ready\",\"protocol\":\"jsonl-v1\","
        <<"\"task\":\"custom-voice\",\"family\":\"1.7b\","
        <<"\"sample_rate\":24000,\"channels\":1,\"sample_format\":\"f32le\","
        <<"\"max_new_tokens\":"<<capacity<<",\"cu_partition\":\""<<partition<<"\","
        <<"\"gpu_name\":\""<<json_escape(properties.name)<<"\","
        <<"\"gpu_arch\":\""<<json_escape(properties.gcnArchName)<<"\","
        <<"\"scheduler_units\":"<<properties.multiProcessorCount<<"}\n"<<std::flush;

    std::string line;
    std::size_t index=0;
    while(std::getline(std::cin,line)) {
        std::uint64_t id=0;
        try {
            if(line.size()>65536) throw std::runtime_error("JSONL request exceeds 64 KiB");
            const auto value=qwen3_tts::baseline::model_frontend::JP(line).parse();
            if(value.kind!=J::K::Object) throw std::runtime_error("request must be an object");
            if(value.o.size()==1 && value.o.count("command") &&
               value.at("command").kind==J::K::String && value.at("command").s=="shutdown") return 0;
            auto integer=[&](const char* key,std::uint64_t minimum,std::uint64_t maximum) {
                const auto& v=value.at(key);
                if(v.kind!=J::K::Number || !std::isfinite(v.n) || std::floor(v.n)!=v.n ||
                   v.n<minimum || v.n>maximum) throw std::runtime_error(std::string("invalid ")+key);
                return static_cast<std::uint64_t>(v.n);
            };
            id=integer("request_id",1,4294967295ULL);
            for(const auto& [key,unused]:value.o) {
                (void)unused;
                if(key!="request_id" && key!="text" && key!="language" && key!="speaker" &&
                   key!="instruct" && key!="output" && key!="seed" && key!="max_new_tokens" && key!="stream_audio")
                    throw std::runtime_error("unsupported CustomVoice request field: "+key);
            }
            auto string=[&](const char* key,bool required) {
                const auto it=value.o.find(key);
                if(it==value.o.end() && !required) return std::string{};
                const auto& v=value.at(key);
                if(v.kind!=J::K::String || (required&&v.s.empty()) || v.s.find('\0')!=std::string::npos)
                    throw std::runtime_error(std::string("invalid ")+key);
                return v.s;
            };
            qwen3_tts::baseline::ResidentRequest request;
            request.text=string("text",true);
            request.language=string("language",true);
            request.speaker=string("speaker",true);
            request.instruct=string("instruct",false);
            request.output_wav=string("output",true);
            request.seed=integer("seed",0,4294967295ULL);
            request.max_new_tokens=integer("max_new_tokens",1,capacity);
            const auto& stream=value.at("stream_audio");
            if(stream.kind!=J::K::Bool) throw std::runtime_error("stream_audio must be boolean");
            qwen3_tts::baseline::codec_decoder::AudioChunkCallback callback;
            if(stream.b) callback=[&audio,id](const float* data,std::size_t count){audio.audio(id,data,count);};
            std::string captured;
            {
                CapturedIo io;
                io.begin();
                engine.generate(request,++index,std::move(callback));
                io.end();
                captured=io.out.str();
            }
            std::string result;
            {
                CapturedIo io;
                io.begin();
                print_resident_result(captured,request,"custom-voice","streaming");
                io.end();
                result=io.out.str();
            }
            while(!result.empty() && std::isspace(static_cast<unsigned char>(result.back()))) result.pop_back();
            const std::string event="{\"event\":\"completed\",\"request_id\":"+std::to_string(id)+
                ",\"result\":"+result+"}";
            std::cout<<event<<"\n"<<std::flush;
            audio.frame(Kind::done,id,event.data(),static_cast<std::uint32_t>(event.size()));
        } catch(const std::exception& error) {
            const std::string event="{\"event\":\"error\",\"request_id\":"+std::to_string(id)+
                ",\"error\":\""+json_escape(error.what())+"\"}";
            std::cout<<event<<"\n"<<std::flush;
            audio.frame(Kind::error,id,event.data(),static_cast<std::uint32_t>(event.size()));
            // Generation failures can leave persistent GPU state incomplete.
            // The owner must restart this worker instead of reusing that state.
            return 1;
        }
    }
    return 0;
}
