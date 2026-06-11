using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
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
#if WINDOWS
        private static readonly object cudaDllLock = new object();
        private static readonly List<IntPtr> loadedCudaLibraries = new List<IntPtr>();
        private static bool cudaDllsPrepared = false;
#endif

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
            lock (cudaDllLock) {
                if (cudaDllsPrepared) {
                    return;
                }

                var candidates = getCudaDllSearchDirectories();
                prependDllSearchDirectoriesToPath(candidates);
                preloadCudaDlls(candidates);
                preloadOrtProviderDlls(candidates);
                cudaDllsPrepared = true;
            }
        }

        private static List<string> getCudaDllSearchDirectories() {
            var candidates = new List<string>();

            void AddDirectory(string? path) {
                if (String.IsNullOrWhiteSpace(path)) {
                    return;
                }
                try {
                    if (Directory.Exists(path)) {
                        candidates.Add(Path.GetFullPath(path));
                    }
                } catch {
                    // Ignore malformed paths.
                }
            }

            void AddCandidate(string? path) {
                if (String.IsNullOrWhiteSpace(path)) {
                    return;
                }
                AddDirectory(path);
                AddDirectory(Path.Combine(path, "bin"));
            }

            void AddCudnnCandidate(string? path) {
                if (String.IsNullOrWhiteSpace(path)) {
                    return;
                }
                AddCandidate(path);
                try {
                    var rootPath = Path.GetFullPath(path);
                    var binPath = Path.Combine(rootPath, "bin");
                    if (Directory.Exists(binPath)) {
                        foreach (var directory in Directory.EnumerateDirectories(binPath, "*", SearchOption.AllDirectories)) {
                            if (Directory.EnumerateFiles(directory, "cudnn*.dll", SearchOption.TopDirectoryOnly).Any()) {
                                AddDirectory(directory);
                            }
                        }
                    }
                    if (Directory.EnumerateFiles(rootPath, "cudnn*.dll", SearchOption.TopDirectoryOnly).Any()) {
                        AddDirectory(rootPath);
                    }
                } catch {
                    // Ignore malformed or inaccessible cuDNN paths.
                }
            }

            AddCandidate(AppContext.BaseDirectory);
            AddCandidate(Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH_V12_4"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH_V12_3"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH_V12_2"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH_V12_1"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH_V12_0"));
            AddCandidate(Environment.GetEnvironmentVariable("CUDA_PATH"));
            AddCudnnCandidate(Environment.GetEnvironmentVariable("CUDNN_PATH"));
            AddCudnnCandidate(Environment.GetEnvironmentVariable("CUDNN_HOME"));
            AddCudnnCandidate(Environment.GetEnvironmentVariable("CUDNN_ROOT"));

            var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            foreach (var cudaVersion in new[] { "v12.4", "v12.3", "v12.2", "v12.1", "v12.0" }) {
                AddCandidate(Path.Combine(programFiles, "NVIDIA GPU Computing Toolkit", "CUDA", cudaVersion));
            }
            AddCudnnCandidate(Path.Combine(programFiles, "NVIDIA", "CUDNN"));
            try {
                var cudnnRoot = Path.Combine(programFiles, "NVIDIA", "CUDNN");
                if (Directory.Exists(cudnnRoot)) {
                    foreach (var cudnnVersionPath in Directory.EnumerateDirectories(cudnnRoot, "v*", SearchOption.TopDirectoryOnly)) {
                        AddCudnnCandidate(cudnnVersionPath);
                    }
                }
            } catch {
                // Ignore malformed or inaccessible default cuDNN paths.
            }

            foreach (var pathEntry in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)) {
                AddCandidate(pathEntry);
                if (pathEntry.Contains("CUDNN", StringComparison.OrdinalIgnoreCase)) {
                    AddCudnnCandidate(pathEntry);
                }
            }

            return candidates
                .Where(Directory.Exists)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
        }

        private static void prependDllSearchDirectoriesToPath(List<string> candidates) {
            var pathEntries = (Environment.GetEnvironmentVariable("PATH") ?? "")
                .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries);
            var knownPathEntries = new HashSet<string>(pathEntries, StringComparer.OrdinalIgnoreCase);
            var newPathEntries = candidates
                .Where(path => !knownPathEntries.Contains(path))
                .ToList();

            if (newPathEntries.Count > 0) {
                Environment.SetEnvironmentVariable("PATH", string.Join(Path.PathSeparator, newPathEntries.Concat(pathEntries)));
                Log.Information("Added CUDA DLL search paths: {CudaPaths}", string.Join(", ", newPathEntries));
            }
        }

        private static void preloadCudaDlls(List<string> candidates) {
            var cudaDlls = new[] {
                "cudart64_12.dll",
                "cublas64_12.dll",
                "cublasLt64_12.dll",
                "curand64_10.dll",
                "cufft64_11.dll",
                "cudnn64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_cnn64_9.dll",
                "cudnn_adv64_9.dll",
                "cudnn_graph64_9.dll",
                "cudnn_engines_precompiled64_9.dll",
                "cudnn_engines_runtime_compiled64_9.dll",
                "cudnn_heuristic64_9.dll",
            };

            foreach (var dllName in cudaDlls) {
                preloadDllIfAvailable(dllName, candidates);
            }
        }

        private static void preloadOrtProviderDlls(List<string> candidates) {
            foreach (var dllName in new[] { "onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll" }) {
                var dllPath = findDll(dllName, candidates);
                if (dllPath == null) {
                    Log.Error("Required ONNX Runtime CUDA provider DLL {OrtProviderDll} was not found. Check the publish output and Microsoft.ML.OnnxRuntime.Gpu package native assets.", dllName);
                    continue;
                }
                try {
                    var handle = NativeLibrary.Load(dllPath);
                    loadedCudaLibraries.Add(handle);
                    Log.Information("Loaded ONNX Runtime provider DLL {OrtProviderDll} from {OrtProviderDllPath}", dllName, dllPath);
                } catch (Exception e) {
                    Log.Error(e, "Failed to preload ONNX Runtime provider DLL {OrtProviderDll} from {OrtProviderDllPath}", dllName, dllPath);
                }
            }
        }

        private static void preloadDllIfAvailable(string dllName, List<string> candidates) {
            var dllPath = findDll(dllName, candidates);
            if (dllPath == null) {
                Log.Warning("{CudaDll} was not found. ONNX Runtime CUDA may fail to initialize if this DLL is required.", dllName);
                return;
            }

            try {
                var handle = NativeLibrary.Load(dllPath);
                loadedCudaLibraries.Add(handle);
                Log.Information("Loaded {CudaDll} from {CudaDllPath}", dllName, dllPath);
            } catch (Exception e) {
                Log.Warning(e, "Failed to preload {CudaDll} from {CudaDllPath}", dllName, dllPath);
            }
        }

        private static string? findDll(string dllName, List<string> candidates) {
            foreach (var directory in candidates) {
                try {
                    var dllPath = Path.Combine(directory, dllName);
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
                    Log.Error(e, "Failed to create ONNX CUDA session. Install CUDA 12.x and cuDNN 9.x, ensure their bin directories are on PATH or copy the required DLLs next to OpenUtau.exe, then restart OpenUtau.");
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
                    Log.Error(e, "Failed to create ONNX CUDA session. Install CUDA 12.x and cuDNN 9.x, ensure their bin directories are on PATH or copy the required DLLs next to OpenUtau.exe, then restart OpenUtau.");
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
