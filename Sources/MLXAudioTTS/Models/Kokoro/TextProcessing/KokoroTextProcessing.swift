//
//  KokoroTextProcessing.swift
//  MLXAudio
//
//  Ported from mlx-audio Kokoro implementation
//

import libespeak_ng
import Foundation
@preconcurrency import NaturalLanguage

// MARK: - ESpeakNGEngine

/// ESpeakNG wrapper for phonemizing the text strings
public final class ESpeakNGEngine {
    private var language: LanguageDialect = .none
    private var languageMapping: [String: String] = [:]

    public enum ESpeakNGEngineError: Error {
        case dataBundleNotFound
        case couldNotInitialize
        case languageNotFound
        case internalError
        case languageNotSet
        case couldNotPhonemize
    }

    public init() throws {
        #if !targetEnvironment(simulator)
        if let bundleURLStr = findDataBundlePath() {
            let initOK = espeak_Initialize(AUDIO_OUTPUT_PLAYBACK, 0, bundleURLStr, 0)

            if initOK != Constants.successAudioSampleRate {
                print("Internal espeak-ng error, could not initialize")
                throw ESpeakNGEngineError.couldNotInitialize
            }

            var languageList: Set<String> = []
            let voiceList = espeak_ListVoices(nil)
            var index = 0
            while let voicePointer = voiceList?.advanced(by: index).pointee {
                let voice = voicePointer.pointee
                if let cLang = voice.languages {
                    let language = String(cString: cLang, encoding: .utf8)!
                        .replacingOccurrences(of: "\u{05}", with: "")
                        .replacingOccurrences(of: "\u{02}", with: "")
                    languageList.insert(language)

                    if let cName = voice.identifier {
                        let name = String(cString: cName, encoding: .utf8)!
                            .replacingOccurrences(of: "\u{05}", with: "")
                            .replacingOccurrences(of: "\u{02}", with: "")
                        languageMapping[language] = name
                    }
                }

                index += 1
            }

            for dialect in LanguageDialect.allCases {
                if dialect.rawValue.count > 0, !languageList.contains(dialect.rawValue) {
                    print("Language dialect \(dialect) not found in espeak-ng voice list")
                    throw ESpeakNGEngineError.languageNotFound
                }
            }
        } else {
            print("Couldn't find the espeak-ng data bundle, cannot initialize")
            throw ESpeakNGEngineError.dataBundleNotFound
        }
        #else
        throw ESpeakNGEngineError.couldNotInitialize
        #endif
    }

    deinit {
        #if !targetEnvironment(simulator)
        let terminateOK = espeak_Terminate()
        print("ESpeakNGEngine termination OK: \(terminateOK == EE_OK)")
        #endif
    }

    public func setLanguage(for voice: KokoroVoice) throws {
        #if !targetEnvironment(simulator)
        let dialect = voice.languageDialect
        guard let name = languageMapping[dialect.rawValue] else {
            throw ESpeakNGEngineError.languageNotFound
        }

        let result = espeak_SetVoiceByName((name as NSString).utf8String)

        if result == EE_NOT_FOUND {
            throw ESpeakNGEngineError.languageNotFound
        } else if result != EE_OK {
            throw ESpeakNGEngineError.internalError
        }

        self.language = dialect
        #else
        throw ESpeakNGEngineError.languageNotFound
        #endif
    }

    public func languageForVoice(voice: KokoroVoice) -> LanguageDialect {
        return voice.languageDialect
    }

    public func phonemize(text: String) throws -> String {
        #if !targetEnvironment(simulator)
        guard language != .none else {
            throw ESpeakNGEngineError.languageNotSet
        }

        guard !text.isEmpty else {
            return ""
        }

        let textCopy = text as NSString
        var textPtr = UnsafeRawPointer(textCopy.utf8String)
        let phonemes_mode = Int32((Int32(Character("_").asciiValue!) << 8) | 0x02)

        let result = autoreleasepool { () -> [String] in
            withUnsafeMutablePointer(to: &textPtr) { ptr in
                var resultWords: [String] = []
                while ptr.pointee != nil {
                    if let result = espeak_TextToPhonemes(ptr, espeakCHARS_UTF8, phonemes_mode) {
                        resultWords.append(String(cString: result, encoding: .utf8)!)
                    }
                }
                return resultWords
            }
        }

        if !result.isEmpty {
            return postProcessPhonemes(result.joined(separator: " "))
        } else {
            throw ESpeakNGEngineError.couldNotPhonemize
        }
        #else
        throw ESpeakNGEngineError.couldNotPhonemize
        #endif
    }

    private func postProcessPhonemes(_ phonemes: String) -> String {
        var result = phonemes.trimmingCharacters(in: .whitespacesAndNewlines)
        for (old, new) in Constants.E2M {
            result = result.replacingOccurrences(of: old, with: new)
        }

        result = result.replacingOccurrences(of: "(\\S)\u{0329}", with: "ᵊ$1", options: .regularExpression)
        result = result.replacingOccurrences(of: "\u{0329}", with: "")

        if language == .enGB {
            result = result.replacingOccurrences(of: "e^ə", with: "ɛː")
            result = result.replacingOccurrences(of: "iə", with: "ɪə")
            result = result.replacingOccurrences(of: "ə^ʊ", with: "Q")
        } else {
            result = result.replacingOccurrences(of: "o^ʊ", with: "O")
            result = result.replacingOccurrences(of: "ɜːɹ", with: "ɜɹ")
            result = result.replacingOccurrences(of: "ɜː", with: "ɜɹ")
            result = result.replacingOccurrences(of: "ɪə", with: "iə")
            result = result.replacingOccurrences(of: "ː", with: "")
        }

        result = result.replacingOccurrences(of: "o", with: "ɔ")
        return result.replacingOccurrences(of: "^", with: "")
    }

    private func findDataBundlePath() -> String? {
        // Try to find espeak-ng-data in various locations
        let executableURL = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
        let cwd = FileManager.default.currentDirectoryPath
        print("[ESpeakNG] Executable URL: \(executableURL.path)")
        print("[ESpeakNG] Current working directory: \(cwd)")
        print("[ESpeakNG] Bundle count: \(Bundle.allBundles.count)")

        // 1. Try Bundle.module for the espeak-ng-data SPM bundle
        //    The bundle name is "espeak-ng_data" (with underscore)
        if let bundleURL = Bundle.allBundles.first(where: { $0.bundlePath.contains("espeak-ng_data") }),
           let dataPath = bundleURL.path(forResource: "espeak-ng-data", ofType: nil) {
            // Return parent directory since espeak_Initialize expects the directory containing espeak-ng-data
            return (dataPath as NSString).deletingLastPathComponent
        }

        // 2. Try finding the bundle by looking in all loaded bundles
        for bundle in Bundle.allBundles {
            if let dataURL = bundle.url(forResource: "espeak-ng-data", withExtension: nil) {
                return dataURL.deletingLastPathComponent().path
            }
        }

        // 3. Try the main bundle directly
        if let mainBundleURL = Bundle.main.url(forResource: "espeak-ng-data", withExtension: nil) {
            return mainBundleURL.deletingLastPathComponent().path
        }

        // 4. Try looking relative to the executable for SPM builds
        let buildDir = executableURL.deletingLastPathComponent()

        // Check for SPM bundle structure (symlinks may be broken, check actual files)
        let spmBundlePath = buildDir.appendingPathComponent("espeak-ng_data.bundle/espeak-ng-data")
        if FileManager.default.fileExists(atPath: spmBundlePath.path) {
            return buildDir.appendingPathComponent("espeak-ng_data.bundle").path
        }

        // 5. For SPM tests, check the current working directory (project root)
        let cwdURL = URL(fileURLWithPath: cwd)
        let cwdCheckoutsPath = cwdURL.appendingPathComponent(".build/checkouts/espeak-ng-spm/Sources/libespeak-ng/_repo/espeak-ng-data")
        print("[ESpeakNG] Checking CWD path: \(cwdCheckoutsPath.path)")
        if FileManager.default.fileExists(atPath: cwdCheckoutsPath.path) {
            print("[ESpeakNG] Found at CWD: \(cwdCheckoutsPath.deletingLastPathComponent().path)")
            return cwdCheckoutsPath.deletingLastPathComponent().path
        }

        // 6. For SPM builds, look in the checkouts directory relative to executable
        //    Navigate from build dir (.build/arm64-apple-macosx/debug) to .build/checkouts
        var searchDir = buildDir
        for i in 0..<5 {
            let checkoutsPath = searchDir.appendingPathComponent("checkouts/espeak-ng-spm/Sources/libespeak-ng/_repo/espeak-ng-data")
            print("[ESpeakNG] Checking (iteration \(i)): \(checkoutsPath.path)")
            if FileManager.default.fileExists(atPath: checkoutsPath.path) {
                // Return the parent directory containing espeak-ng-data
                print("[ESpeakNG] Found at: \(checkoutsPath.deletingLastPathComponent().path)")
                return checkoutsPath.deletingLastPathComponent().path
            }
            // Also check .build directory at this level
            let buildCheckoutsPath = searchDir.appendingPathComponent(".build/checkouts/espeak-ng-spm/Sources/libespeak-ng/_repo/espeak-ng-data")
            if FileManager.default.fileExists(atPath: buildCheckoutsPath.path) {
                print("[ESpeakNG] Found at: \(buildCheckoutsPath.deletingLastPathComponent().path)")
                return buildCheckoutsPath.deletingLastPathComponent().path
            }
            searchDir = searchDir.deletingLastPathComponent()
        }

        // 6. Try loading the bundle manually by path
        let bundlePath = buildDir.appendingPathComponent("espeak-ng_data.bundle")
        if FileManager.default.fileExists(atPath: bundlePath.path) {
            if let dataBundle = Bundle(url: bundlePath) {
                dataBundle.load()
                if let dataPath = dataBundle.path(forResource: "espeak-ng-data", ofType: nil) {
                    return (dataPath as NSString).deletingLastPathComponent
                }
            }
        }

        // 7. Legacy: Try framework bundle identifier
        if let frameworkBundle = Bundle(identifier: "com.kokoro.espeakng"),
           let dataBundleURL = frameworkBundle.url(forResource: "espeak-ng-data", withExtension: "bundle")
        {
            return dataBundleURL.path
        }

        return nil
    }

    private enum Constants {
        static let successAudioSampleRate: Int32 = 22050
        static let E2M: [(String, String)] = [
            ("ʔˌn\u{0329}", "tn"), ("ʔn\u{0329}", "tn"), ("ʔn", "tn"), ("ʔ", "t"),
            ("a^ɪ", "I"), ("a^ʊ", "W"),
            ("d^ʒ", "ʤ"),
            ("e^ɪ", "A"), ("e", "A"),
            ("t^ʃ", "ʧ"),
            ("ɔ^ɪ", "Y"),
            ("ə^l", "ᵊl"),
            ("ʲo", "jo"), ("ʲə", "jə"), ("ʲ", ""),
            ("ɚ", "əɹ"),
            ("r", "ɹ"),
            ("x", "k"), ("ç", "k"),
            ("ɐ", "ə"),
            ("ɬ", "l"),
            ("\u{0303}", ""),
        ].sorted(by: { $0.0.count > $1.0.count })
    }
}

// MARK: - PhonemeTokenizer

/// Utility class for tokenizing the phonemized text
public final class PhonemeTokenizer {
    private init() {}

    public static func tokenize(phonemizedText text: String) -> [Int] {
        return text
            .map { Constants.vocab[String($0)] }
            .filter { $0 != nil }
            .map { $0! }
    }

    private enum Constants {
        static let vocab: [String: Int] = [
            ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, "\"": 11, "(": 12,
            ")": 13, "\u{201C}": 14, "\u{201D}": 15, " ": 16, "\u{0303}": 17, "ʣ": 18, "ʥ": 19, "ʦ": 20,
            "ʨ": 21, "ᵝ": 22, "\u{AB67}": 23, "A": 24, "I": 25, "O": 31, "Q": 33, "S": 35,
            "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44, "c": 45, "d": 46, "e": 47,
            "f": 48, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54, "m": 55, "n": 56, "o": 57,
            "p": 58, "q": 59, "r": 60, "s": 61, "t": 62, "u": 63, "v": 64, "w": 65, "x": 66,
            "y": 67, "z": 68, "ɑ": 69, "ɐ": 70, "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77,
            "ç": 78, "ɖ": 80, "ð": 81, "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90,
            "ɡ": 92, "ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111, "ŋ": 112, "ɳ": 113,
            "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123, "ɾ": 125,
            "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132, "ʧ": 133, "ʊ": 135,
            "ʋ": 136, "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148,
            "ˈ": 156, "ˌ": 157, "ː": 158, "ʰ": 162, "ʲ": 164, "↓": 169, "→": 171, "↗": 172,
            "↘": 173, "ᵻ": 177,
        ]
    }
}

// MARK: - SentenceTokenizer

public final class SentenceTokenizer {
    private init() {}

    public static func splitIntoSentences(text: String) -> [String] {
        guard !text.isEmpty else { return [] }

        let detectedLanguage = detectLanguage(text: text)
        let initialSentences = performInitialSplit(text: text, language: detectedLanguage)
        let refinedSentences = applyTTSRefinements(sentences: initialSentences, originalText: text)

        return optimizeTTSChunks(sentences: refinedSentences, language: detectedLanguage)
    }

    private static func performInitialSplit(text: String, language: NLLanguage?) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text

        if let language = language {
            tokenizer.setLanguage(language)
        }

        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { tokenRange, _ in
            let sentence = String(text[tokenRange])
            sentences.append(sentence)
            return true
        }

        return sentences.isEmpty ? [text] : sentences
    }

    private static func applyTTSRefinements(sentences: [String], originalText: String) -> [String] {
        var result: [String] = []
        result.reserveCapacity(sentences.count)

        for sentence in sentences {
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                result.append(trimmed)
            }
        }

        return result
    }

    private static func optimizeTTSChunks(sentences: [String], language: NLLanguage?) -> [String] {
        guard !sentences.isEmpty else { return [] }

        let scriptType = detectScriptType(language: language)

        switch scriptType {
        case .cjk:
            return optimizeCJKChunks(sentences: sentences)
        case .indic:
            return optimizeIndicChunks(sentences: sentences)
        case .latin, .other:
            return optimizeLatinChunks(sentences: sentences)
        }
    }

    private static func optimizeLatinChunks(sentences: [String]) -> [String] {
        let minLength = 50
        return optimizeChunks(
            sentences: sentences,
            config: ChunkConfig(
                minLength: minLength,
                maxLength: 300,
                separator: " ",
                shouldMerge: { chunk in
                    chunk.count < minLength || !hasStrongSentenceEnding(chunk)
                }
            )
        )
    }

    private static func optimizeCJKChunks(sentences: [String]) -> [String] {
        let minLength = 30
        return optimizeChunks(
            sentences: sentences,
            config: ChunkConfig(
                minLength: minLength,
                maxLength: 200,
                separator: "",
                shouldMerge: { chunk in
                    chunk.count < minLength || !hasCJKSentenceEnding(chunk)
                }
            )
        )
    }

    private static func optimizeIndicChunks(sentences: [String]) -> [String] {
        let minLength = 40
        return optimizeChunks(
            sentences: sentences,
            config: ChunkConfig(
                minLength: minLength,
                maxLength: 250,
                separator: " ",
                shouldMerge: { chunk in
                    chunk.count < minLength || !hasIndicSentenceEnding(chunk)
                }
            )
        )
    }

    private struct ChunkConfig {
        let minLength: Int
        let maxLength: Int
        let separator: String
        let shouldMerge: (String) -> Bool
    }

    private static func optimizeChunks(sentences: [String], config: ChunkConfig) -> [String] {
        guard !sentences.isEmpty else { return [] }

        var result: [String] = []
        result.reserveCapacity(sentences.count)
        var currentChunk = ""

        for sentence in sentences {
            if currentChunk.isEmpty {
                currentChunk = sentence
            } else {
                let separatorLength = config.separator.isEmpty ? 0 : config.separator.count
                let potentialLength = currentChunk.count + sentence.count + separatorLength

                if potentialLength <= config.maxLength && config.shouldMerge(currentChunk) {
                    if !config.separator.isEmpty {
                        currentChunk += config.separator
                    }
                    currentChunk += sentence
                } else {
                    result.append(currentChunk)
                    currentChunk = sentence
                }
            }
        }

        if !currentChunk.isEmpty {
            result.append(currentChunk)
        }

        return result
    }

    private static let languageRecognizer = NLLanguageRecognizer()

    private static func detectLanguage(text: String) -> NLLanguage? {
        languageRecognizer.reset()
        languageRecognizer.processString(text)
        return languageRecognizer.dominantLanguage
    }

    private enum ScriptType {
        case latin, cjk, indic, other
    }

    private static func detectScriptType(language: NLLanguage?) -> ScriptType {
        guard let language = language else { return .other }

        switch language {
        case .simplifiedChinese, .traditionalChinese, .japanese:
            return .cjk
        case .english, .french, .spanish, .italian, .portuguese:
            return .latin
        case .hindi:
            return .indic
        default:
            return .other
        }
    }

    private static func hasSentenceEnding(_ text: String, endings: Set<Character>) -> Bool {
        guard let lastChar = text.last else { return false }
        return endings.contains(lastChar) && !text.hasSuffix(" ")
    }

    private static func hasStrongSentenceEnding(_ text: String) -> Bool {
        return hasSentenceEnding(text, endings: [".", "!", "?"])
    }

    private static func hasCJKSentenceEnding(_ text: String) -> Bool {
        return hasSentenceEnding(text, endings: ["。", "！", "？", "…"])
    }

    private static func hasIndicSentenceEnding(_ text: String) -> Bool {
        return hasSentenceEnding(text, endings: ["।", "॥", ".", "!", "?"])
    }
}

// MARK: - KokoroTokenizer

/// Converts text into phonetic representations with stress markers for on-device text-to-speech
public final class KokoroTokenizer {

    private static let stresses = "ˌˈ"
    private static let primaryStress = Character("ˈ")
    private static let secondaryStress = Character("ˌ")

    private static let vowels: Set<Character> = [
        "A", "I", "O", "Q", "W", "Y", "a", "i", "u",
        "æ", "ɑ", "ɒ", "ɔ", "ə", "ɛ", "ɜ", "ɪ", "ʊ", "ʌ", "ᵻ"
    ]

    private static let subtokenJunks: Set<Character> = [
        "'", ",", "-", ".", "_", "'", "'", "/", " "
    ]

    private static let puncts: Set<String> = [
        "?", ",", ";", "\"", "—", ":", "!", ".", "…", "\""
    ]

    private static let functionWords: Set<String> = [
        "a", "an", "the", "in", "on", "at", "of", "for", "with", "by", "to", "from",
        "and", "or", "but", "nor", "so", "yet", "is", "am", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "shall", "should", "may", "might", "must", "can", "could", "that",
        "this", "these", "those", "he", "she", "it", "they", "we", "you", "i", "me",
        "him", "her", "them", "us", "my", "your", "his", "their", "our", "its"
    ]

    private static let sentenceEndingPunct: Set<String> = [".", "!", "?"]

    private static let currencyRegex = try! NSRegularExpression(
        pattern: #"[\$£€]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[\$£€]\d+\.\d\d?\b"#
    )

    private static let timeRegex = try! NSRegularExpression(
        pattern: #"\b(?:[1-9]|1[0-2]):[0-5]\d\b"#
    )

    private static let decimalRegex = try! NSRegularExpression(
        pattern: #"\b\d*\.\d+\b"#
    )

    private static let rangeRegex = try! NSRegularExpression(
        pattern: #"([\$£€]?\d+)-([\$£€]?\d+)"#
    )

    private static let commaInNumberRegex = try! NSRegularExpression(
        pattern: #"(^|[^\d])(\d+(?:,\d+)*)([^\d]|$)"#
    )

    private static let linkRegex = try! NSRegularExpression(
        pattern: #"\[([^\]]+)\]\(([^\)]*)\)"#
    )

    private static let alphabeticRegex = try! NSRegularExpression(
        pattern: #"^[a-zA-Z]+$"#
    )

    private static let currencies: [Character: (bill: String, cent: String)] = [
        "$": ("dollar", "cent"),
        "£": ("pound", "pence"),
        "€": ("euro", "cent")
    ]

    private var cachedUSLexicon: [String: String]?
    private var cachedGBLexicon: [String: String]?
    private var eSpeakEngine: ESpeakNGEngine
    private var currentLanguage: LanguageDialect = .none
    private var isLexiconEnabled = false

    public struct Token {
        public let text: String
        public var whitespace: String
        public var phonemes: String
        public let stress: Float?
        public let currency: String?
        public var prespace: Bool
        public let alias: String?
        public let isHead: Bool

        public init(
            text: String,
            whitespace: String = " ",
            phonemes: String = "",
            stress: Float? = nil,
            currency: String? = nil,
            prespace: Bool = true,
            alias: String? = nil,
            isHead: Bool = false
        ) {
            self.text = text
            self.whitespace = whitespace
            self.phonemes = phonemes
            self.stress = stress
            self.currency = currency
            self.prespace = prespace
            self.alias = alias
            self.isHead = isHead
        }
    }

    public struct PhonemizerResult {
        public let phonemes: String
        public let tokens: [Token]
    }

    public init(engine: ESpeakNGEngine) {
        self.eSpeakEngine = engine
        loadLexicon()
    }

    public func setLanguage(for voice: KokoroVoice) throws {
        let language = eSpeakEngine.languageForVoice(voice: voice)
        if currentLanguage != language {
            try eSpeakEngine.setLanguage(for: voice)
            currentLanguage = language

            switch language {
            case .enUS, .enGB:
                self.isLexiconEnabled = true
            default:
                self.isLexiconEnabled = false
            }
        }
    }

    public func phonemize(_ text: String) throws -> PhonemizerResult {
        let (_, tokens, features, nonStringFeatures) = preprocess(text)
        let tokenizedTokens = try tokenize(tokens: tokens, features: features, nonStringFeatures: nonStringFeatures)
        let result = resolveTokens(tokenizedTokens)
        return PhonemizerResult(phonemes: result, tokens: tokenizedTokens)
    }

    private func loadLexicon() {
        var usLexicon: [String: String] = [:]
        if let silverLexicon = loadLexiconFile("us_silver") {
            usLexicon.merge(silverLexicon) { _, new in new }
        }
        if let goldLexicon = loadLexiconFile("us_gold") {
            usLexicon.merge(goldLexicon) { _, new in new }
        }
        self.cachedUSLexicon = usLexicon.isEmpty ? nil : usLexicon

        var gbLexicon: [String: String] = [:]
        if let silverLexicon = loadLexiconFile("gb_silver") {
            gbLexicon.merge(silverLexicon) { _, new in new }
        }
        if let goldLexicon = loadLexiconFile("gb_gold") {
            gbLexicon.merge(goldLexicon) { _, new in new }
        }
        self.cachedGBLexicon = gbLexicon.isEmpty ? nil : gbLexicon
    }

    private func loadLexiconFile(_ filename: String) -> [String: String]? {
        guard let url = Bundle.module.url(forResource: filename, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        var processedLexicon: [String: String] = [:]

        for (key, value) in json {
            if let stringValue = value as? String {
                processedLexicon[key] = stringValue
            } else if let dictValue = value as? [String: Any],
                      let defaultValue = dictValue["DEFAULT"] as? String
            {
                processedLexicon[key] = defaultValue
            }
        }

        return processedLexicon
    }

    private func preprocess(_ text: String) -> (String, [String], [String: Any], Set<String>) {
        var processedText = text
        var tokens: [String] = []
        var features: [String: Any] = [:]
        var nonStringFeatures: Set<String> = []

        processedText = removeCommasFromNumbers(processedText)

        processedText = Self.rangeRegex.stringByReplacingMatches(
            in: processedText,
            range: NSRange(processedText.startIndex..., in: processedText),
            withTemplate: "$1 to $2"
        )

        processedText = flipMoney(processedText)
        processedText = splitNum(processedText)
        processedText = pointNum(processedText)

        var lastEnd = processedText.startIndex
        var result = ""

        let matches = Self.linkRegex.matches(
            in: processedText,
            range: NSRange(processedText.startIndex..., in: processedText)
        )

        for match in matches {
            let beforeMatch = String(processedText[lastEnd..<processedText.index(processedText.startIndex, offsetBy: match.range.location)])
            result += beforeMatch
            tokens.append(contentsOf: beforeMatch.components(separatedBy: " ").filter { !$0.isEmpty })

            let originalRange = Range(match.range(at: 1), in: processedText)!
            let replacementRange = Range(match.range(at: 2), in: processedText)!
            let original = String(processedText[originalRange])
            let replacement = String(processedText[replacementRange])

            let (feature, isNonString) = parseFeature(original: original, replacement: replacement)
            let tokenIndex = String(tokens.count)

            if isNonString {
                nonStringFeatures.insert(tokenIndex)
            }
            features[tokenIndex] = feature

            result += original
            tokens.append(original)

            lastEnd = processedText.index(processedText.startIndex, offsetBy: match.range.upperBound)
        }

        if lastEnd < processedText.endIndex {
            let remaining = String(processedText[lastEnd...])
            result += remaining
            tokens.append(contentsOf: remaining.components(separatedBy: " ").filter { !$0.isEmpty })
        }

        return (result, tokens, features, nonStringFeatures)
    }

    private func parseFeature(original: String, replacement: String) -> (Any, Bool) {
        if replacement.hasPrefix("/") && replacement.hasSuffix("/") {
            return (replacement, false)
        }

        if original.hasPrefix("$") || original.contains(":") || original.contains(".") {
            return ("[\(replacement)]", false)
        }

        if let value = Float(replacement) {
            return (value, true)
        }

        if replacement.hasPrefix("-") || replacement.hasPrefix("+") {
            if let value = Float(replacement) {
                return (value, true)
            }
        }

        return ("[\(replacement)]", false)
    }

    private func removeCommasFromNumbers(_ text: String) -> String {
        return Self.commaInNumberRegex.stringByReplacingMatches(
            in: text,
            range: NSRange(text.startIndex..., in: text),
            withTemplate: "$1$2$3"
        ).replacingOccurrences(of: ",", with: "")
    }

    private func flipMoney(_ text: String) -> String {
        var result = text
        let matches = Self.currencyRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        for match in matches.reversed() {
            let matchRange = Range(match.range, in: text)!
            let matchText = String(text[matchRange])

            guard let currencySymbol = matchText.first,
                  let currency = Self.currencies[currencySymbol]
            else { continue }

            let value = String(matchText.dropFirst())
            let components = value.components(separatedBy: ".")
            let dollars = components[0]
            let cents = components.count > 1 ? components[1] : "0"

            let transformed: String
            if Int(cents) == 0 {
                transformed = Int(dollars) == 1 ? "\(dollars) \(currency.bill)" : "\(dollars) \(currency.bill)s"
            } else {
                let dollarPart = Int(dollars) == 1 ? "\(dollars) \(currency.bill)" : "\(dollars) \(currency.bill)s"
                transformed = "\(dollarPart) and \(cents) \(currency.cent)s"
            }

            result = result.replacingCharacters(in: matchRange, with: "\(transformed)")
        }

        return result
    }

    private func splitNum(_ text: String) -> String {
        var result = text
        let matches = Self.timeRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        for match in matches.reversed() {
            let matchRange = Range(match.range, in: text)!
            let matchText = String(text[matchRange])

            let components = matchText.components(separatedBy: ":")
            guard components.count == 2,
                  let hour = Int(components[0]),
                  let minute = Int(components[1])
            else { continue }

            let transformed: String
            if minute == 0 {
                transformed = "\(hour) o'clock"
            } else if minute < 10 {
                transformed = "\(hour) oh \(minute)"
            } else {
                transformed = "\(hour) \(minute)"
            }

            result = result.replacingCharacters(in: matchRange, with: "\(transformed)")
        }

        return result
    }

    private func pointNum(_ text: String) -> String {
        var result = text
        let decimalMatches = Self.decimalRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        let linkMatches = Self.linkRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        var excludeRanges: [NSRange] = []
        for linkMatch in linkMatches {
            if linkMatch.numberOfRanges >= 3 {
                let replacementRange = linkMatch.range(at: 2)
                excludeRanges.append(replacementRange)
            }
        }

        for match in decimalMatches.reversed() {
            let matchRange = match.range
            let isInLinkReplacement = excludeRanges.contains { NSIntersectionRange(matchRange, $0).length > 0 }
            if isInLinkReplacement { continue }

            let swiftRange = Range(matchRange, in: text)!
            let matchText = String(text[swiftRange])

            let components = matchText.components(separatedBy: ".")
            guard components.count == 2 else { continue }

            let integerPart = components[0]
            let decimalPart = components[1]
            let decimalDigits = decimalPart.map { String($0) }.joined(separator: " ")
            let transformed = "\(integerPart) point \(decimalDigits)"

            result = result.replacingCharacters(in: swiftRange, with: "\(transformed)")
        }

        return result
    }

    private func tokenize(tokens: [String], features: [String: Any], nonStringFeatures: Set<String>) throws -> [Token] {
        var result: [Token] = []

        for (index, word) in tokens.enumerated() {
            if word.contains(Self.subtokenJunks) { continue }

            let feature = features[String(index)]

            if let featureString = feature as? String,
               featureString.hasPrefix("/") && featureString.hasSuffix("/")
            {
                let phoneme = String(featureString.dropFirst().dropLast())
                result.append(Token(text: word, phonemes: phoneme))
                continue
            }

            if feature is Float {
                let wordTokens = try tokenizeWord(word, stress: feature as? Float)
                if !wordTokens.isEmpty {
                    result.append(mergeTokens(wordTokens))
                }
                continue
            }

            var phonemeText = word
            if let featureString = feature as? String,
               featureString.hasPrefix("[") && featureString.hasSuffix("]")
            {
                phonemeText = String(featureString.dropFirst().dropLast())
            }

            let wordTokens = try tokenizeWord(phonemeText, stress: feature as? Float)
            if !wordTokens.isEmpty {
                result.append(mergeTokens(wordTokens))
            }
        }

        return result
    }

    private func tokenizeWord(_ text: String, stress: Float?) throws -> [Token] {
        let words = text.components(separatedBy: " ").filter { !$0.isEmpty }
        var tokens: [Token] = []

        for (_, word) in words.enumerated() {
            let punctSplit = splitPunctuation(word)

            for (subIndex, token) in punctSplit.enumerated() {
                let phoneme = try generatePhoneme(for: token)
                let isWhitespace = !Self.puncts.contains(token)

                if !isWhitespace && !tokens.isEmpty {
                    tokens[tokens.count - 1].whitespace = ""
                }

                tokens.append(Token(
                    text: token,
                    phonemes: phoneme,
                    stress: stress,
                    prespace: isWhitespace,
                    isHead: subIndex == 0
                ))
            }
        }

        if !tokens.isEmpty {
            tokens[tokens.count - 1].whitespace = ""
            tokens[0].prespace = false
        }

        return tokens
    }

    private func splitPunctuation(_ text: String) -> [String] {
        var result = [text]

        for punct in Self.puncts {
            var newResult: [String] = []
            for part in result {
                if part.contains(punct) {
                    newResult.append(contentsOf: part.components(separatedBy: punct).flatMap { [$0, punct] }.dropLast())
                } else {
                    newResult.append(part)
                }
            }
            result = newResult.filter { !$0.isEmpty }
        }

        return result
    }

    private func generatePhoneme(for token: String) throws -> String {
        if Self.puncts.contains(token) {
            return token
        }

        let lowerToken = token.lowercased()

        if isLexiconEnabled {
            let lexicon: [String: String]?
            switch currentLanguage {
            case .enUS:
                lexicon = cachedUSLexicon
            case .enGB:
                lexicon = cachedGBLexicon
            default:
                lexicon = nil
            }

            if let selectedLexicon = lexicon,
               let phoneme = selectedLexicon[token] ?? selectedLexicon[lowerToken]
            {
                return phoneme
            }
        }

        return try eSpeakEngine.phonemize(text: lowerToken)
    }

    private func mergeTokens(_ tokens: [Token]) -> Token {
        let stresses = tokens.compactMap { $0.stress }

        var phonemes = ""
        for token in tokens {
            if token.prespace && !phonemes.isEmpty && !phonemes.last!.isWhitespace && !token.phonemes.isEmpty {
                phonemes += " "
            }
            phonemes += token.phonemes
        }

        if phonemes.first?.isWhitespace == true {
            phonemes = String(phonemes.dropFirst())
        }

        let mergedStress = stresses.count == 1 ? stresses[0] : nil
        let text = tokens.dropLast().map { $0.text + $0.whitespace }.joined() + tokens.last!.text

        return Token(
            text: text,
            whitespace: tokens.last?.whitespace ?? "",
            phonemes: phonemes,
            stress: mergedStress,
            prespace: tokens.first?.prespace ?? false,
            isHead: tokens.first?.isHead ?? false
        )
    }

    private func resolveTokens(_ tokens: [Token]) -> String {
        let phonemeCorrections: [String: String] = [
            "eɪ": "A",
            "ɹeɪndʒ": "ɹAnʤ",
            "wɪðɪn": "wəðɪn"
        ]

        let wordPhonemeMap: [String: String] = [
            "a": "ɐ",
            "an": "ən"
        ]

        var processedTokens = tokens

        for (index, token) in processedTokens.enumerated() {
            guard !token.phonemes.isEmpty else { continue }

            if let mapped = wordPhonemeMap[token.text.lowercased()] {
                processedTokens[index] = Token(
                    text: token.text,
                    whitespace: token.whitespace,
                    phonemes: mapped,
                    stress: token.stress,
                    currency: token.currency,
                    prespace: token.prespace,
                    alias: token.alias,
                    isHead: token.isHead
                )
                continue
            }

            var correctedPhonemes = token.phonemes
            for (old, new) in phonemeCorrections {
                correctedPhonemes = correctedPhonemes.replacingOccurrences(of: old, with: new)
            }

            if let customStress = token.stress {
                correctedPhonemes = applyCustomStress(to: correctedPhonemes, stressValue: customStress)
            } else {
                let hasStress = correctedPhonemes.contains(Self.primaryStress) || correctedPhonemes.contains(Self.secondaryStress)

                if !hasStress {
                    if correctedPhonemes.contains(" ") {
                        let subwords = correctedPhonemes.components(separatedBy: " ")
                        let stressedSubwords = subwords.map { subword -> String in
                            guard !subword.isEmpty && !subword.contains(Self.primaryStress) && !subword.contains(Self.secondaryStress) else {
                                return subword
                            }

                            let hasVowels = subword.contains { Self.vowels.contains($0) }
                            guard hasVowels else { return subword }

                            if ["ænd", "ðə", "ɪn", "ɔn", "æt", "wɪð", "baɪ"].contains(subword) {
                                return subword
                            } else {
                                return addStressBeforeVowel(subword, stress: Self.primaryStress)
                            }
                        }
                        correctedPhonemes = stressedSubwords.joined(separator: " ")
                    } else {
                        if index == 0 {
                            correctedPhonemes = addStressBeforeVowel(correctedPhonemes, stress: Self.secondaryStress)
                        } else if isContentWord(token.text) && correctedPhonemes.count > 2 {
                            correctedPhonemes = addStressBeforeVowel(correctedPhonemes, stress: Self.primaryStress)
                        }
                    }
                }
            }

            processedTokens[index] = Token(
                text: token.text,
                whitespace: token.whitespace,
                phonemes: correctedPhonemes,
                stress: token.stress,
                currency: token.currency,
                prespace: token.prespace,
                alias: token.alias,
                isHead: token.isHead
            )
        }

        var result: [String] = []
        var punctuationAdded = false

        for (index, token) in processedTokens.enumerated() {
            let isPunct = Self.puncts.contains(token.text)

            if index > 0 && !isPunct && !punctuationAdded {
                result.append(" ")
            }

            punctuationAdded = false

            if isPunct {
                result.append(token.text)
                punctuationAdded = true
            } else if !token.phonemes.isEmpty {
                result.append(token.phonemes)

                if token.text.last.map({ Self.puncts.contains(String($0)) }) == true {
                    let punct = String(token.text.last!)
                    result.append(punct)
                    punctuationAdded = true

                    if Self.sentenceEndingPunct.contains(punct) && index < processedTokens.count - 1 {
                        result.append(" ")
                    }
                }
            }
        }

        return result.joined()
    }

    private func addStressBeforeVowel(_ phoneme: String, stress: Character) -> String {
        for (index, char) in phoneme.enumerated() {
            if Self.vowels.contains(char) {
                if index == 0 {
                    return String(stress) + phoneme
                } else {
                    let insertIndex = phoneme.index(phoneme.startIndex, offsetBy: index)
                    return String(phoneme[..<insertIndex]) + String(stress) + String(phoneme[insertIndex...])
                }
            }
        }
        return phoneme
    }

    private func applyCustomStress(to phonemes: String, stressValue: Float) -> String {
        var result = phonemes

        result = result.replacingOccurrences(of: String(Self.primaryStress), with: "")
        result = result.replacingOccurrences(of: String(Self.secondaryStress), with: "")

        if stressValue < -1 {
            return result
        } else if stressValue == -1 {
            return addStressBeforeVowel(result, stress: Self.secondaryStress)
        } else if stressValue >= 0 && stressValue < 1 {
            return addStressBeforeVowel(result, stress: Self.secondaryStress)
        } else if stressValue >= 1 {
            return addStressBeforeVowel(result, stress: Self.primaryStress)
        }

        return result
    }

    private func isFunctionWord(_ word: String) -> Bool {
        let cleaned = word.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: String(Self.puncts.joined())))
        return Self.functionWords.contains(cleaned)
    }

    private func isContentWord(_ word: String) -> Bool {
        return !isFunctionWord(word) && word.count > 2 && isAlphabetic(word)
    }

    private func isAlphabetic(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        return Self.alphabeticRegex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }
}

private extension Character {
    var isWhitespace: Bool {
        return self == " " || self == "\t" || self == "\n" || self == "\r"
    }
}

private extension String {
    func contains(_ set: Set<Character>) -> Bool {
        return self.contains { set.contains($0) }
    }
}
