using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core {
    public class GpuInfo {
        public int deviceId;
        public string description = "";

        override public string ToString() {
            return $"[{deviceId}] {description}";
        }
    }

    public enum OnnxRunnerChoice {
        Default,
        CPU,
        CPUForCoreML,
    }

    public class Onnx {
        private static readonly Dictionary<int, OrtEpDevice> devices = initializeDevices();

        private static Dictionary<int, OrtEpDevice> initializeDevices() {
            try {
                var env = OrtEnv.Instance();
                var ortDevices = env.GetEpDevices();

                return ortDevices
                    .Where(device => device.EpName.ToLower().Contains("dml"))
                    .Select((device, index) => new { index, device })
                    .ToDictionary(x => x.index, x => x.device);
            } catch (Exception e) {
                Log.Warning(e, "Failed to enumerate ONNX Runtime execution provider devices");
                return new Dictionary<int, OrtEpDevice>();
            }
        }

        public static List<string> getRunnerOptions() {
            if (OS.IsWindows()) {
                return new List<string> {
                "CUDA",
                "CPU"
                };
            } else if (OS.IsMacOS()) {
                return new List<string> {
                "CPU",
                "CoreML"
                };
            } else if (OS.IsAndroid()) {
                return new List<string> {
                "CPU",
                "NNAPI"
                };
            }
            return new List<string> {
                "CPU"
            };
        }

        public static List<GpuInfo> getGpuInfo() {
            if (OS.IsAndroid()) {
                return new List<GpuInfo>{new GpuInfo {
                    deviceId = 0, // eliminate exception of taking OnnxGpuOptions[0]
                }};
            }
            if (OS.IsWindows() && getSelectedRunner() == "CUDA") {
                return new List<GpuInfo>{new GpuInfo {
                    deviceId = 0,
                    description = "CUDA GPU 0",
                }};
            }
            List<GpuInfo> gpuList = new List<GpuInfo>();
            var env = OrtEnv.Instance();
            var ortDevices = env.GetEpDevices();

            var i = 0;
            foreach (var device in ortDevices.Where(device => device.EpName.ToLower().Contains("dml"))) {
                var description = "";
                foreach (var item in device.HardwareDevice.Metadata.Entries) {
                    if (item.Key.ToLower() == "description") {
                        description = $"{item.Value} ({device.HardwareDevice.Type})";
                        break;
                    }
                }
                if (string.IsNullOrEmpty(description)) { // fallback
                    description = $"{device.EpName} {device.HardwareDevice.Vendor} ({device.HardwareDevice.Type})";
                }
                devices[i] = device;
                gpuList.Add(new GpuInfo {
                    deviceId = i++,
                    description = description
                });
            }
            if (gpuList.Count == 0) {
                gpuList.Add(new GpuInfo {
                    deviceId = 0,
                });
            }
            return gpuList;
        }

        private static string getSelectedRunner() {
            List<string> runnerOptions = getRunnerOptions();
            string runner = Preferences.Default.OnnxRunner;
            if (OS.IsWindows() && (String.IsNullOrEmpty(runner) || runner == "DirectML")) {
                runner = "CUDA";
            }
            if (String.IsNullOrEmpty(runner)) {
                runner = runnerOptions[0];
            }
            if (!runnerOptions.Contains(runner)) {
                runner = "CPU";
            }
            return runner;
        }

#if WINDOWS
        private static SessionOptions getCudaSessionOptions() {
            prepareCudaDllSearchPath();
            var deviceId = Math.Max(0, Preferences.Default.OnnxGpu);
            Log.Information("Creating ONNX Runtime CUDA session on device {CudaDeviceId}", deviceId);
            return SessionOptions.MakeSessionOptionWithCudaProvider(deviceId);
        }

        private static void prepareCudaDllSearchPath() {
            var candidates = new List<string>();

            void AddCandidate(string? path) {
                if (String.IsNullOrWhiteSpace(path)) {
                    return;
                }
                if (Directory.Exists(path)) {
                    candidates.Add(path);
                }
                var binPath = Path.Combine(path, "bin");
                if (Directory.Exists(binPath)) {
                    candidates.Add(binPath);
                }
            }

            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH_V12_4"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH_V12_3"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH_V12_2"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH_V12_1"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH_V12_0"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDNN_PATH"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDNN_HOME"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDNN_ROOT"));

            var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            foreach (var cudaVersion in new[] { "v12.4", "v12.3", "v12.2", "v12.1", "v12.0" }) {
                AddCandidate(Path.Combine(programFiles, "NVIDIA GPU Computing Toolkit", "CUDA", cudaVersion));
            }

            var pathEntries = (Environment.GetEnvironmentVariable("PATH") ?? "")
                .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries);
            var knownPathEntries = new HashSet<string>(pathEntries, StringComparer.OrdinalIgnoreCase);
            var newPathEntries = candidates
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Where(path => !knownPathEntries.Contains(path))
                .ToList();

            if (newPathEntries.Count > 0) {
                Environment.SetEnvironmentVariable("PATH", string.Join(Path.PathSeparator, newPathEntries.Concat(pathEntries)));
                Log.Information("Added CUDA DLL search paths: {CudaPaths}", string.Join(", ", newPathEntries));
            }

            logDllAvailability("cudart64_12.dll");
            logDllAvailability("cudnn64_9.dll");
        }

        private static void logDllAvailability(string dllName) {
            var dllPath = findDllOnPath(dllName);
            if (dllPath == null) {
                Log.Warning("{CudaDll} was not found on PATH. ONNX Runtime CUDA may fail to initialize and require CUDA 12.x/cuDNN 9.x runtime DLLs.", dllName);
            } else {
                Log.Information("Found {CudaDll} at {CudaDllPath}", dllName, dllPath);
            }
        }

        private static string? findDllOnPath(string dllName) {
            foreach (var pathEntry in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)) {
                try {
                    var dllPath = Path.Combine(pathEntry, dllName);
                    if (File.Exists(dllPath)) {
                        return dllPath;
                    }
                } catch {
                    // Ignore malformed PATH entries.
                }
            }
            return null;
        }
#endif

        private static SessionOptions getOnnxSessionOptions(bool coremlEnableOnSubgraphs = false) {
            string runner = getSelectedRunner();
            switch (runner) {
#if WINDOWS
                case "CUDA":
                    return getCudaSessionOptions();
#endif
                case "DirectML":
                    var d = devices[Preferences.Default.OnnxGpu];
                    var dmlOptions = new SessionOptions();
                    dmlOptions.AppendExecutionProvider(
                        OrtEnv.Instance(),
                        new List<OrtEpDevice> { d },
                        new Dictionary<string, string> { }
                     );
                    return dmlOptions;
                case "CoreML":
                    var coremlOptions = new SessionOptions();
                    // Note: MLProgram format has stricter validation and may fail with complex DiffSinger models
                    // that have topological sorting issues (e.g., variance_predictor with diffusion embeddings)
                    // so we always use NeuralNetwork format (default) as MLProgram fails with complex models.
                    coremlOptions.AppendExecutionProvider("CoreML", new Dictionary<string, string> {
                        { "MLComputeUnits", "ALL" },
                        { "RequireStaticInputShapes", "1"},
                        { "ModelFormat", "NeuralNetwork"},
                        { "EnableOnSubgraphs", coremlEnableOnSubgraphs ? "1" : "0" }  // Disable subgraph processing to avoid complex control flow issues
                    });
                    return coremlOptions;
                case "NNAPI":
                    var nnapiOptions = new SessionOptions();
                    nnapiOptions.AppendExecutionProvider_Nnapi();
                    return nnapiOptions;
            }
            return new SessionOptions();
        }

        public static InferenceSession getInferenceSession(byte[] model, OnnxRunnerChoice runnerChoice = OnnxRunnerChoice.Default) {
            string runner = getSelectedRunner();
            if (runnerChoice == OnnxRunnerChoice.CPU ||
                (runnerChoice == OnnxRunnerChoice.CPUForCoreML && runner == "CoreML")) {
                return new InferenceSession(model);
            } else {
                // Try with CoreML subgraphs enabled first, fallback to default if it fails
                if (OS.IsMacOS() && runner == "CoreML") {
                    try {
                        return new InferenceSession(model, getOnnxSessionOptions(coremlEnableOnSubgraphs: true));
                    } catch (Exception e) {
                        Log.Warning(e, "Failed to create session with CoreML subgraphs enabled, falling back to default settings");
                    }
                }
                try {
                    return new InferenceSession(model, getOnnxSessionOptions());
                } catch (Exception e) when (runner == "CUDA") {
                    Log.Error(e, "Failed to create ONNX CUDA session. Install CUDA 12.x and cuDNN 9.x, ensure their bin directories are on PATH, then restart OpenUtau.");
                    throw;
                } catch (Exception e) when (runner != "CPU") {
                    Log.Warning(e, "Failed to create ONNX session with {OnnxRunner}; falling back to CPU", runner);
                    return new InferenceSession(model);
                }
            }
        }

        public static InferenceSession getInferenceSession(string modelPath, OnnxRunnerChoice runnerChoice = OnnxRunnerChoice.Default) {
            string runner = getSelectedRunner();
            if (runnerChoice == OnnxRunnerChoice.CPU ||
                (runnerChoice == OnnxRunnerChoice.CPUForCoreML && runner == "CoreML")) {
                return new InferenceSession(modelPath);
            } else {
                // Try with CoreML subgraphs enabled first, fallback to default if it fails
                if (OS.IsMacOS() && runner == "CoreML") {
                    try {
                        return new InferenceSession(modelPath, getOnnxSessionOptions(coremlEnableOnSubgraphs: true));
                    } catch (Exception e) {
                        Log.Warning(e, "Failed to create session with CoreML subgraphs enabled, falling back to default settings");
                    }
                }
                try {
                    return new InferenceSession(modelPath, getOnnxSessionOptions());
                } catch (Exception e) when (runner == "CUDA") {
                    Log.Error(e, "Failed to create ONNX CUDA session. Install CUDA 12.x and cuDNN 9.x, ensure their bin directories are on PATH, then restart OpenUtau.");
                    throw;
                } catch (Exception e) when (runner != "CPU") {
                    Log.Warning(e, "Failed to create ONNX session with {OnnxRunner}; falling back to CPU", runner);
                    return new InferenceSession(modelPath);
                }
            }
        }

        public static void VerifyInputNames(InferenceSession session, IEnumerable<NamedOnnxValue> inputs) {
            var sessionInputNames = session.InputNames.ToHashSet();
            var givenInputNames = inputs.Select(v => v.Name).ToHashSet();
            var missing = sessionInputNames
                .Except(givenInputNames)
                .OrderBy(s => s, StringComparer.InvariantCulture)
                .ToArray();
            if (missing.Length > 0) {
                throw new ArgumentException("Missing input(s) for the inference session: " + string.Join(", ", missing));
            }
            var unexpected = givenInputNames
                .Except(sessionInputNames)
                .OrderBy(s => s, StringComparer.InvariantCulture)
                .ToArray();
            if (unexpected.Length > 0) {
                throw new ArgumentException("Unexpected input(s) for the inference session: " + string.Join(", ", unexpected));
            }
        }
    }
}
