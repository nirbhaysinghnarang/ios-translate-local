import SwiftUI
import AVFoundation
import MLKitTranslate

// MARK: - Models
struct TranscriptMessage: Identifiable {
    let id = UUID()
    let sourceText: String
    var translatedText: String?
    let timestamp: Date
    let sourceLang: Language
    let targetLang: Language
}

struct Language: Identifiable, Hashable {
    let id = UUID()
    let code: String
    let name: String
    
    static let supported: [Language] = [
        Language(code: "af", name: "Afrikaans"),
        Language(code: "ar", name: "Arabic"),
        Language(code: "bn", name: "Bengali"),
        Language(code: "zh", name: "Chinese"),
        Language(code: "en", name: "English"),
        Language(code: "fr", name: "French"),
        Language(code: "de", name: "German"),
        Language(code: "hi", name: "Hindi"),
        Language(code: "id", name: "Indonesian"),
        Language(code: "it", name: "Italian"),
        Language(code: "ja", name: "Japanese"),
        Language(code: "ko", name: "Korean"),
        Language(code: "pt", name: "Portuguese"),
        Language(code: "ru", name: "Russian"),
        Language(code: "es", name: "Spanish"),
        Language(code: "sw", name: "Swahili"),
        Language(code: "tr", name: "Turkish"),
        Language(code: "vi", name: "Vietnamese")
    ]
}

// MARK: - Views
struct LoadingView: View {
    let message: String
    
    var body: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(1.5)
            Text(message)
                .font(.headline)
                .foregroundColor(.gray)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground).opacity(0.9))
    }
}

struct MessageView: View {
    let message: TranscriptMessage
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(message.sourceLang.name)
                    .font(.caption)
                    .foregroundColor(.gray)
                Spacer()
                Text(message.timestamp.formatted(date: .omitted, time: .shortened))
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            
            Text(message.sourceText)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            if let translatedText = message.translatedText {
                Divider()
                HStack {
                    Text(message.targetLang.name)
                        .font(.caption)
                        .foregroundColor(.gray)
                    Spacer()
                }
                Text(translatedText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .foregroundColor(.blue)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

// MARK: - Audio Processing Manager
class AudioProcessingManager: ObservableObject {
    private var audioEngine: AVAudioEngine?
    private var processor: WhisperVADProcessor?
    private var translator: Translator?
    
    @Published var isRecording = false
    @Published var messages: [TranscriptMessage] = []
    @Published var currentInterimText: String = ""
    @Published var currentTranslatedInterim: String = ""
    @Published var isLoadingModels = false
    @Published var loadingMessage = ""
    @Published var isDownloadingTranslation = false
    
    private var selectedSourceLang: Language
    private var selectedTargetLang: Language
    private var translationWorkItem: DispatchWorkItem?
    
    init(sourceLang: Language, targetLang: Language) {
        self.selectedSourceLang = sourceLang
        self.selectedTargetLang = targetLang
        setupAudioEngine()
        setupProcessor()
        initializeTranslator()
    }
    
    private func setupAudioEngine() {
        audioEngine = AVAudioEngine()
    }
    
    private func initializeTranslator() {
        let options = TranslatorOptions(
            sourceLanguage: getMLKitLanguage(from: selectedSourceLang.code),
            targetLanguage: getMLKitLanguage(from: selectedTargetLang.code)
        )
        translator = Translator.translator(options: options)
        
        let conditions = ModelDownloadConditions(
            allowsCellularAccess: true,
            allowsBackgroundDownloading: true
        )
        
        isDownloadingTranslation = true
        translator?.downloadModelIfNeeded(with: conditions) { [weak self] error in
            DispatchQueue.main.async {
                if let error = error {
                    print("Translation model download error: \(error)")
                    self?.isDownloadingTranslation = false
                    return
                }
                print("Translation model downloaded successfully")
                self?.isDownloadingTranslation = false
            }
        }
    }
    
    private func getMLKitLanguage(from code: String) -> TranslateLanguage {
        switch code {
        case "af": return .afrikaans
        case "ar": return .arabic
        case "bn": return .bengali
        case "zh": return .chinese
        case "en": return .english
        case "fr": return .french
        case "de": return .german
        case "hi": return .hindi
        case "id": return .indonesian
        case "it": return .italian
        case "ja": return .japanese
        case "ko": return .korean
        case "pt": return .portuguese
        case "ru": return .russian
        case "es": return .spanish
        case "sw": return .swahili
        case "tr": return .turkish
        case "vi": return .vietnamese
        default: return .english
        }
    }
    
    private func translate(_ text: String, isInterim: Bool = false, completion: ((String?) -> Void)? = nil) {
        translator?.translate(text) { [weak self] translatedText, error in
            guard let translatedText = translatedText else {
                completion?(nil)
                return
            }
            
            DispatchQueue.main.async {
                if isInterim {
                    self?.currentTranslatedInterim = translatedText
                }
                completion?(translatedText)
            }
        }
    }
    
    private func setupProcessor() {
        isLoadingModels = true
        loadingMessage = "Loading AI Models..."
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            let whisperModel = getWhisperModel(sourceLanguage: self.selectedSourceLang.code)
            
            self.loadingMessage = "Initializing Voice Detection..."
            
            let vadModelConfig = sherpaOnnxSileroVadModelConfig(
                model: Bundle.main.path(forResource: "silero_vad", ofType: "onnx") ?? ""
            )
            var vadConfig = sherpaOnnxVadModelConfig(sileroVad: vadModelConfig)
            var vad = SherpaOnnxVoiceActivityDetectorWrapper(
                config: &vadConfig,
                buffer_size_in_seconds: 5.0
            )
            
            DispatchQueue.main.async {
                self.loadingMessage = "Configuring Speech Processor..."
            }
            
            self.processor = WhisperVADProcessor(
                whisperModel: whisperModel,
                vad: vad,
                processInterim: { [weak self] interimText in
                    DispatchQueue.main.async {
                        self?.currentInterimText = interimText
                        // Translate interim text with debounce
                        self?.translationWorkItem?.cancel()
                        let workItem = DispatchWorkItem { [weak self] in
                            self?.translate(interimText, isInterim: true)
                        }
                        self?.translationWorkItem = workItem
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3, execute: workItem)
                    }
                },
                processFinal: { [weak self] finalText in
                    guard let self = self else { return }
                    // Translate final text
                    self.translate(finalText) { translatedText in
                        DispatchQueue.main.async {
                            let message = TranscriptMessage(
                                sourceText: finalText,
                                translatedText: translatedText,
                                timestamp: Date(),
                                sourceLang: self.selectedSourceLang,
                                targetLang: self.selectedTargetLang
                            )
                            self.messages.append(message)
                            self.currentInterimText = ""
                            self.currentTranslatedInterim = ""
                        }
                    }
                }
            )
            
            DispatchQueue.main.async {
                self.isLoadingModels = false
            }
        }
    }
    
    func updateLanguages(source: Language, target: Language) {
        selectedSourceLang = source
        selectedTargetLang = target
        if isRecording {
            stopRecording()
        }
        setupProcessor()
        initializeTranslator()
    }
    
    func startRecording() throws {
        guard let audioEngine = audioEngine else { return }
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0,
                           bufferSize: 1024,
                           format: recordingFormat) { [weak self] buffer, _ in
            self?.processor?.processAudio(buffer)
        }
        
        loadingMessage = "Starting Audio Engine..."
        audioEngine.prepare()
        try audioEngine.start()
        
        DispatchQueue.main.async { [weak self] in
            self?.isRecording = true
            self?.isLoadingModels = false
            self?.loadingMessage = ""
        }
    }
    
    func stopRecording() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        processor?.pauseTranscribing()
        isRecording = false
        currentInterimText = ""
        currentTranslatedInterim = ""
    }
}

// MARK: - Content View
struct ContentView: View {
    @State private var selectedSourceLang = Language.supported.first(where: { $0.code == "en" })!
    @State private var selectedTargetLang = Language.supported.first(where: { $0.code == "es" })!
    @StateObject private var audioManager: AudioProcessingManager
    @State private var showingError = false
    @State private var errorMessage = ""
    @State private var scrollProxy: ScrollViewProxy?
    
    init() {
        let initialSource = Language.supported.first(where: { $0.code == "en" })!
        let initialTarget = Language.supported.first(where: { $0.code == "es" })!
        _audioManager = StateObject(wrappedValue: AudioProcessingManager(
            sourceLang: initialSource,
            targetLang: initialTarget
        ))
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) { // Removed spacing here
                // Top Controls
                VStack(spacing: 8) {
                    // Language Selection
                    HStack {
                        Picker("From", selection: $selectedSourceLang) {
                            ForEach(Language.supported) { language in
                                Text(language.name).tag(language)
                            }
                        }
                        .pickerStyle(.menu)
                        
                        Image(systemName: "arrow.right")
                            .foregroundColor(.gray)
                        
                        Picker("To", selection: $selectedTargetLang) {
                            ForEach(Language.supported) { language in
                                Text(language.name).tag(language)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    .padding(.horizontal)
                    .padding(.top, 8)
                }
                .background(Color(.systemBackground))
                .onChange(of: selectedSourceLang) { _ in updateLanguages() }
                .onChange(of: selectedTargetLang) { _ in updateLanguages() }
                
                // Messages List - Now takes up most of the space
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(audioManager.messages) { message in
                                MessageView(message: message)
                                    .id(message.id)
                            }
                            
                            if !audioManager.currentInterimText.isEmpty {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text(audioManager.currentInterimText)
                                        .italic()
                                        .foregroundColor(.gray)
                                    
                                    if !audioManager.currentTranslatedInterim.isEmpty {
                                        Divider()
                                        Text(audioManager.currentTranslatedInterim)
                                            .italic()
                                            .foregroundColor(.blue)
                                    }
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding()
                                .background(Color(.systemGray6))
                                .cornerRadius(10)
                                .id("interim")
                            }
                        }
                        .padding()
                    }
                    .onChange(of: audioManager.messages.count) { _ in
                        withAnimation {
                            proxy.scrollTo(audioManager.messages.last?.id.uuidString ?? "interim", anchor: .bottom)
                        }
                    }
                    .onChange(of: audioManager.currentInterimText) { _ in
                        withAnimation {
                            proxy.scrollTo("interim", anchor: .bottom)
                        }
                    }
                }
                
                // Bottom Control Bar
                VStack(spacing: 0) {
                    Divider()
                    
                    // Record Button
                    Button(action: toggleRecording) {
                        VStack(spacing: 4) {
                            Image(systemName: audioManager.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                                .font(.system(size: 44))
                                .foregroundColor(audioManager.isRecording ? .red : .blue)
                            Text(audioManager.isRecording ? "Stop" : "Start")
                                .font(.caption)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(Color(.systemBackground))
                }
            }
            .navigationTitle("Live Translation")
            .navigationBarTitleDisplayMode(.inline)
            .overlay {
                if audioManager.isLoadingModels || audioManager.isDownloadingTranslation {
                    LoadingView(message: audioManager.isDownloadingTranslation ?
                              "Downloading translation model..." : audioManager.loadingMessage)
                }
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    private func toggleRecording() {
        if audioManager.isRecording {
            audioManager.stopRecording()
        } else {
            do {
                try audioManager.startRecording()
            } catch {
                errorMessage = error.localizedDescription
                showingError = true
            }
        }
    }
    
    private func updateLanguages() {
        audioManager.updateLanguages(source: selectedSourceLang, target: selectedTargetLang)
    }
}
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
