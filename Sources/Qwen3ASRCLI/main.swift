import Foundation
import Qwen3ASR
import Qwen3Common

/// Resolve a model specifier to a HuggingFace model ID.
/// Accepts shorthand ("0.6B", "1.7B") or full model IDs.
func resolveModelId(_ specifier: String) -> String {
    switch specifier.lowercased() {
    case "0.6b", "small":
        return ASRModelSize.small.defaultModelId
    case "1.7b", "large":
        return ASRModelSize.large.defaultModelId
    default:
        return specifier
    }
}

// Entry point
Task {
    let args = CommandLine.arguments

    // Parse arguments
    var audioPath: String?
    var modelId = ASRModelSize.small.defaultModelId

    var i = 1
    while i < args.count {
        switch args[i] {
        case "--model", "-m":
            i += 1
            guard i < args.count else {
                print("Error: --model requires a value")
                print("  Examples: --model 1.7B, --model mlx-community/Qwen3-ASR-1.7B-8bit")
                exit(1)
            }
            modelId = resolveModelId(args[i])
        case "--help", "-h":
            print("Usage: qwen3-asr-cli [options] <audio-file>")
            print()
            print("Options:")
            print("  --model, -m <model>  Model to use (default: 0.6B)")
            print("                       Shorthand: 0.6B, 1.7B, small, large")
            print("                       Or full HuggingFace model ID")
            print("  --help, -h           Show this help message")
            print()
            print("Examples:")
            print("  qwen3-asr-cli audio.wav")
            print("  qwen3-asr-cli --model 1.7B audio.wav")
            print("  qwen3-asr-cli -m mlx-community/Qwen3-ASR-1.7B-8bit audio.wav")
            exit(0)
        default:
            if args[i].hasPrefix("-") {
                print("Error: Unknown option '\(args[i])'")
                print("Use --help for usage information")
                exit(1)
            }
            audioPath = args[i]
        }
        i += 1
    }

    guard let audioPath else {
        print("Usage: qwen3-asr-cli [options] <audio-file>")
        print("Use --help for more information")
        exit(1)
    }

    let detectedSize = ASRModelSize.detect(from: modelId)
    let sizeLabel = detectedSize == .large ? "1.7B" : "0.6B"
    print("Loading model (\(sizeLabel)): \(modelId)")

    do {
        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: modelId
        ) { progress, status in
            print("  [\(Int(progress * 100))%] \(status)")
        }

        print("Loading audio: \(audioPath)")
        let audio = try AudioFileLoader.load(url: URL(fileURLWithPath: audioPath), targetSampleRate: 24000)
        print("  Loaded \(audio.count) samples (\(String(format: "%.2f", Double(audio.count) / 24000.0))s)")

        print("Transcribing...")
        let result = model.transcribe(
            audio: audio,
            sampleRate: 24000
        )
        print("Result: \(result)")
        exit(0)
    } catch {
        print("Error: \(error)")
        exit(1)
    }
}

// Keep the main thread alive
RunLoop.main.run()
