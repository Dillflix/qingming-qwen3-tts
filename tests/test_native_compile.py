"""Host-only compile checks: actual IPC glue/parser, stub engine; no HIP qualification."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CXX = os.environ.get("QINGMING_TEST_CXX") or shutil.which("c++") or shutil.which("cl")

PREFIX = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
constexpr int hipSuccess=0;
struct hipDeviceProp_t { const char* name="stub 7900 XT"; const char* gcnArchName="gfx1100"; int multiProcessorCount=42; };
int hipGetDevice(int* p){*p=0;return 0;}
int hipGetDeviceProperties(hipDeviceProp_t*,int){return 0;}
namespace qingming::audio_protocol {
enum class Kind {audio=1,done=2,error=3};
struct Writer {
    std::vector<Kind> kinds;
    void frame(Kind k,std::uint64_t,const void*,std::uint32_t){kinds.push_back(k);}
    void audio(std::uint64_t,const float*,std::size_t){kinds.push_back(Kind::audio);}
};
}
namespace qwen3_tts::baseline {
bool g_resident_cu_partition_enabled=false;
namespace codec_decoder {using AudioChunkCallback=std::function<void(const float*,std::size_t)>;}
struct ResidentRequest {
    std::string text,language,speaker,instruct;
    fs::path output_wav;
    std::uint64_t seed=0;
    std::size_t max_new_tokens=0;
};
struct ResidentEngine {
    int calls=0;
    std::string last_text;
    void generate(const ResidentRequest& r,std::size_t,codec_decoder::AudioChunkCallback callback){
        ++calls; last_text=r.text;
        const float wave[]={0.25f,-0.25f};
        if(callback) callback(wave,2);
        std::cout<<"resident_codec_frame_count: 1\nresident_e2e_ms: 10\nresident_request_success: True\n";
    }
};
namespace model_frontend {
'''

SUFFIX = r'''
int main(){
    using namespace qingming::production;
    qwen3_tts::baseline::ResidentEngine engine;
    qingming::audio_protocol::Writer audio;
    const std::string request=R"({"request_id":1,"text":"Hello \"speaker\": literal text","language":"English","speaker":"Ryan","instruct":"Calm.","output":"test.wav","seed":1234,"max_new_tokens":512,"stream_audio":true})";
    std::istringstream in(request+"\n"+R"({"command":"shutdown"})"+"\n");
    auto* old=std::cin.rdbuf(in.rdbuf());
    assert(run_customvoice_jsonl(engine,audio,512,"none")==0);
    std::cin.rdbuf(old);
    assert(engine.calls==1);
    assert(engine.last_text=="Hello \"speaker\": literal text");
    assert(audio.kinds.size()==2);
    assert(audio.kinds[0]==qingming::audio_protocol::Kind::audio);
    assert(audio.kinds[1]==qingming::audio_protocol::Kind::done);
    std::string bad=request;
    bad.insert(1,"\"ref_audio\":\"forbidden.wav\",");
    std::istringstream invalid(bad+"\n");
    old=std::cin.rdbuf(invalid.rdbuf());
    assert(run_customvoice_jsonl(engine,audio,512,"none")==1);
    std::cin.rdbuf(old);
    assert(engine.calls==1);
    assert(audio.kinds.back()==qingming::audio_protocol::Kind::error);
}
'''


@unittest.skipUnless(CXX, "C++20 host compiler required (QINGMING_TEST_CXX)")
class NativeCompileTests(unittest.TestCase):
    def compile_and_run(self, source):
        with tempfile.TemporaryDirectory(prefix="qingming-native-test-") as directory:
            path = Path(directory)
            cpp = path / "test.cpp"
            cpp.write_text(source, encoding="utf-8")
            exe = path / ("test.exe" if os.name == "nt" else "test")
            if Path(CXX).name.lower() in ("cl", "cl.exe"):
                args = [CXX, "/nologo", "/std:c++20", "/EHsc", "/utf-8", f"/I{ROOT}", str(cpp), f"/Fe{exe}", f"/Fo{path / 'test.obj'}"]
            else:
                args = [CXX, "-std=c++20", "-Wall", "-Wextra", f"-I{ROOT}", str(cpp), "-o", str(exe)]
            result = subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=path)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

    def test_frame_header(self):
        self.compile_and_run(r'''
#include "native/audio_protocol.h"
#include <cassert>
int main(){
    using namespace qingming::audio_protocol;
    const auto h=header(Kind::audio,0x1122334455667788ULL,4096);
    assert(h.size()==24 && h[0]=='Q' && h[3]=='1');
    assert(h[4]==1 && h[8]==0x88 && h[15]==0x11);
    assert(h[16]==0 && h[17]==0x10 && h[20]==0);
    bool rejected=false;
    try{header(Kind::audio,1,max_payload+1);}catch(...){rejected=true;}
    assert(rejected);
}
''')

    def test_offline_speaker_embedding_io(self):
        self.compile_and_run(r'''
#include "native/speaker_embedding.h"
#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
namespace fs=std::filesystem;
namespace embedding=qingming::speaker_embedding;
void rejects(const std::function<void()>& action){
    bool failed=false;
    try{action();}catch(const std::exception&){failed=true;}
    assert(failed);
}
int main(){
    const auto root=fs::temp_directory_path()/(
        "qingming-embedding-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    assert(fs::create_directory(root));
    try{
        const auto profile=root/"speaker with spaces.bf16";
        std::vector<std::uint16_t> values(embedding::dimensions,0x3f80);
        values[1]=0xbf80; values[2]=0x0001; values[3]=0x7f7f;
        embedding::write_exclusive(profile,values);
        assert(fs::file_size(profile)==4096);
        assert(embedding::read(profile)==values);
        std::ifstream raw(profile,std::ios::binary);
        unsigned char bytes[4]{};
        raw.read(reinterpret_cast<char*>(bytes),4); raw.close();
        assert(bytes[0]==0x80 && bytes[1]==0x3f && bytes[2]==0x80 && bytes[3]==0xbf);
        auto replacement=values; replacement[0]=0x4000;
        rejects([&]{embedding::write_exclusive(profile,replacement);});
        assert(embedding::read(profile)==values);
        rejects([&]{embedding::read(root);});
        rejects([&]{embedding::read(root/"missing");});
        rejects([&]{embedding::write_exclusive(root/"missing"/"out",values);});
        rejects([&]{embedding::read(fs::path{});});
        rejects([&]{embedding::read(fs::path(std::string("bad\0path",8)));});
        for(const auto size:{0,1,4095,4097,8192}){
            const auto malformed=root/("size-"+std::to_string(size));
            std::ofstream file(malformed,std::ios::binary);
            std::vector<char> payload(size,0); file.write(payload.data(),payload.size()); file.close();
            rejects([&]{embedding::read(malformed);});
        }
        const auto invalid=root/"invalid.bf16";
        auto bad=values; bad.pop_back();
        rejects([&]{embedding::write_exclusive(invalid,bad);});
        assert(!fs::exists(invalid));
        for(const std::uint16_t zero:{0x0000,0x8000}){
            bad.assign(embedding::dimensions,zero);
            rejects([&]{embedding::write_exclusive(invalid,bad);});
            assert(!fs::exists(invalid));
        }
        for(const std::uint16_t bits:{0x7f80,0xff80,0x7fc1}){
            bad=values; bad[30]=bits;
            rejects([&]{embedding::write_exclusive(invalid,bad);});
            assert(!fs::exists(invalid));
            const auto malformed=root/("nonfinite-"+std::to_string(bits));
            fs::copy_file(profile,malformed);
            std::fstream file(malformed,std::ios::binary|std::ios::in|std::ios::out);
            const char encoded[]={static_cast<char>(bits&0xff),static_cast<char>(bits>>8)};
            file.write(encoded,2); file.close();
            rejects([&]{embedding::read(malformed);});
        }
        embedding::validate_sources("ref.wav",{},profile);
        embedding::validate_sources({},profile,{});
        rejects([&]{embedding::validate_sources({},{},{});});
        rejects([&]{embedding::validate_sources("ref.wav",profile,{});});
        rejects([&]{embedding::validate_sources({},profile,"export");});
        embedding::validate_audio_output(profile,{},root/"out.wav");
        rejects([&]{embedding::validate_audio_output(profile,{},profile);});
        rejects([&]{embedding::validate_audio_output({},profile,root/"."/profile.filename());});
        const auto hardlink=root/"audio-hardlink.wav";
        std::error_code link_error;
        fs::create_hard_link(profile,hardlink,link_error);
        if(!link_error){
            rejects([&]{embedding::validate_audio_output(profile,{},hardlink);});
            assert(embedding::read(profile)==values);
        }
#ifndef _WIN32
        const auto link=root/"link";
        fs::create_symlink(profile,link);
        rejects([&]{embedding::write_exclusive(link,values);});
        assert(embedding::read(profile)==values);
        const auto dangling=root/"dangling";
        fs::create_symlink(root/"absent",dangling);
        rejects([&]{embedding::write_exclusive(dangling,values);});
        assert(!fs::exists(root/"absent"));
#endif
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";fs::remove_all(root);return 1;}
    fs::remove_all(root);
}
''')

    def test_actual_ipc_glue_with_stub_hip_engine(self):
        backend = (ROOT / "devices/rx7900xtx-24g/qwen3_tts_1_7b.cpp").read_text(encoding="utf-8")
        parser = backend[backend.index("struct J {"):backend.index("static std::uint64_t le64(")]
        main = (ROOT / "main.cpp").read_text(encoding="utf-8")
        helpers = main[main.index("struct CapturedIo {"):main.index('#include "native/customvoice_resident.h"')]
        helpers = helpers[:helpers.index("static int run_once(")]
        source = PREFIX + parser + "\n}}\nnamespace qingming::production {\n" + helpers
        source += '\n#include "native/customvoice_resident.h"\n}\n' + SUFFIX
        output = self.compile_and_run(source)
        # The native control stream must remain JSONL, including completed events.
        import json
        events = [json.loads(line) for line in output.splitlines()]
        self.assertEqual([event["event"] for event in events], ["ready", "completed", "ready", "error"])

    def test_actual_once_cli_accepts_offline_base_profiles_only(self):
        backend = (ROOT / "devices/rx7900xtx-24g/qwen3_tts_1_7b.cpp").read_text(encoding="utf-8")
        options = backend[backend.index("struct CliOptions {"):backend.index("static void write_text_file(")]
        parser = backend[backend.index("static CliOptions parse_cli("):backend.index("static int run_self_test()")]
        source = r'''
#include "native/speaker_embedding.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
namespace fs=std::filesystem;
'''
        source += options + parser + r'''
CliOptions parse(std::vector<std::string> extra){
    std::vector<std::string> args={"tts","--model-dir","base-model","--text","Hello",
        "--text-mode","streaming","--max-new-tokens","256"};
    args.insert(args.end(),extra.begin(),extra.end());
    std::vector<char*> argv;
    for(auto& arg:args) argv.push_back(arg.data());
    return parse_cli(static_cast<int>(argv.size()),argv.data());
}
void rejects(std::vector<std::string> args){
    bool failed=false;
    try{parse(std::move(args));}catch(const std::exception&){failed=true;}
    assert(failed);
}
int main(){
    const auto enrolled=parse({"--task","base-xvector","--ref-audio","ref.wav",
        "--save-speaker-embedding-bf16","speaker.bf16"});
    assert(enrolled.ref_audio=="ref.wav" && enrolled.save_speaker_embedding_bf16=="speaker.bf16");
    const auto imported=parse({"--task","base-xvector","--speaker-embedding-bf16","speaker.bf16"});
    assert(imported.ref_audio.empty() && imported.speaker_embedding_bf16=="speaker.bf16");
    rejects({"--task","base-xvector"});
    rejects({"--task","base-xvector","--ref-audio","ref.wav","--speaker-embedding-bf16","speaker.bf16"});
    rejects({"--task","base-xvector","--speaker-embedding-bf16","speaker.bf16","--save-speaker-embedding-bf16","copy.bf16"});
    rejects({"--task","base-xvector","--ref-audio","ref.wav","--instruct","Calm"});
    rejects({"--task","base-xvector","--ref-audio","ref.wav","--save-speaker-embedding-bf16"});
    const auto preset=parse({"--task","custom-voice","--speaker","Aiden","--instruct","Calm"});
    assert(preset.speaker=="Aiden" && preset.instruct=="Calm");
    rejects({"--task","custom-voice","--speaker","Aiden","--speaker-embedding-bf16","speaker.bf16"});
    rejects({"--task","custom-voice","--speaker","Aiden","--save-speaker-embedding-bf16","speaker.bf16"});
    rejects({"--task","voice-design","--instruct","Calm","--save-speaker-embedding-bf16","speaker.bf16"});
}
'''
        self.compile_and_run(source)


if __name__ == "__main__":
    unittest.main()
