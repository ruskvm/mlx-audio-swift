import Foundation
import Hub
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXNN

/// Wrapper that adapts Kokoro to the SpeechGenerationModel protocol
public final class KokoroTTSModel: SpeechGenerationModel {

    private let kokoro: Kokoro
    public let sampleRate: Int

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters()
    }

    public init(kokoro: Kokoro) {
        self.kokoro = kokoro
        self.sampleRate = kokoro.config.sampleRate
    }

    /// Load a Kokoro model from a HuggingFace repo
    public static func fromPretrained(
        _ modelRepo: String,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> KokoroTTSModel {
        let kokoro = try await Kokoro.fromHub(
            repoId: modelRepo,
            progressHandler: progressHandler
        )
        return KokoroTTSModel(kokoro: kokoro)
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        // Resolve voice: use provided voice name or default
        let resolvedVoice = resolveVoice(from: voice, language: language)

        // Extract speed from generation parameters or default to 1.0
        let speed: Float = 1.0

        return try kokoro.generateAudioForSentence(
            voice: resolvedVoice,
            text: text,
            speed: speed
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let resolvedVoice = resolveVoice(from: voice, language: language)
        let kokoro = self.kokoro

        return AsyncThrowingStream { continuation in
            Task { @Sendable in
                do {
                    var chunks: [MLXArray] = []
                    try kokoro.generateAudio(voice: resolvedVoice, text: text, speed: 1.0) { chunk in
                        chunks.append(chunk)
                        continuation.yield(.audio(chunk))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Resolve a voice name string to a KokoroVoice enum value
    private func resolveVoice(from voiceName: String?, language: String?) -> KokoroVoice {
        // If a voice name is provided, try to match it
        if let name = voiceName, !name.isEmpty {
            // Try exact match by fileName
            if let match = KokoroVoice.allCases.first(where: { $0.fileName == name }) {
                return match
            }
            // Try case-insensitive match
            let lower = name.lowercased()
            if let match = KokoroVoice.allCases.first(where: { $0.fileName.lowercased() == lower }) {
                return match
            }
            // Try matching by raw value
            if let match = KokoroVoice(rawValue: name) {
                return match
            }
        }

        // If language is specified, pick a default voice for that language
        if let lang = language?.lowercased() {
            switch lang {
            case "de", "german", "deutsch":
                return .efDora  // European Female (closest to German)
            case "ja", "japanese":
                return .jfAlpha
            case "zh", "chinese", "mandarin":
                return .zfXiaobei
            case "fr", "french":
                return .ffSiwis
            case "ko", "korean":
                return .hfAlpha
            case "hi", "hindi":
                return .hfAlpha
            case "pt", "portuguese":
                return .pfDora
            case "it", "italian":
                return .ifSara
            case "en-gb", "british":
                return .bfAlice
            default:
                break
            }
        }

        // Default to American English female voice
        return .afHeart
    }
}
