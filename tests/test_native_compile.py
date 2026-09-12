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


if __name__ == "__main__":
    unittest.main()
