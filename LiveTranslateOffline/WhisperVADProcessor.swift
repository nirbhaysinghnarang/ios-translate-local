import Foundation
import AVFoundation

class WhisperVADProcessor {
    // MARK: - Constants
    private let MAX_SPEECH_SECS = 15.0
    private let MIN_REFRESH_SECS = 0.1
    private let SAMPLING_RATE = 16000
    private let LOOKBACK_CHUNKS = 40
    private let CHUNK_SIZE = 512
    
    // MARK: - Models
    private var whisperModel: SherpaOnnxOfflineRecognizer
    private var vad: SherpaOnnxVoiceActivityDetectorWrapper
    
    // MARK: - Audio Processing State
    private var samplesBuffer: [Float] = []  // Rolling buffer for lookback
    private var speechBuffer: [Float] = []   // Current speech segment
    private var isRecording = false
    private var recordingStartTime: CFTimeInterval?
    private var lastInterimProcessingTime: CFTimeInterval?
    
    // MARK: - Transcript State
    private var previousInterim: String = ""
    private let processInterim: (String) -> Void
    private let processFinal: (String) -> Void
    
    // MARK: - Initialization
    init(whisperModel: SherpaOnnxOfflineRecognizer,
         vad: SherpaOnnxVoiceActivityDetectorWrapper,
         processInterim: @escaping (String) -> Void,
         processFinal: @escaping (String) -> Void) {
        self.whisperModel = whisperModel
        self.vad = vad
        self.processInterim = processInterim
        self.processFinal = processFinal
    }
    
    // MARK: - Audio Processing
    func processAudio(_ buffer: AVAudioPCMBuffer) {
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(SAMPLING_RATE),
            channels: 1,
            interleaved: false
        ) else { return }
        
        let converter = AVAudioConverter(from: buffer.format, to: outputFormat)!
        var newBufferAvailable = true
        let inputCallback: AVAudioConverterInputBlock = { inNumPackets, outStatus in
            if newBufferAvailable {
                outStatus.pointee = .haveData
                newBufferAvailable = false
                return buffer
            } else {
                outStatus.pointee = .noDataNow
                return nil
            }
        }
        
        guard let convertedBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(outputFormat.sampleRate) * buffer.frameLength / AVAudioFrameCount(buffer.format.sampleRate)
        ) else { return }
        
        var error: NSError?
        converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputCallback)
        var array = convertedBuffer.array()
        guard !array.isEmpty else { return }
        
        while !array.isEmpty {
            let chunkSize = min(CHUNK_SIZE, array.count)
            var chunk = Array(array[0..<chunkSize])
            if chunk.count < CHUNK_SIZE {
                chunk.append(contentsOf: Array(repeating: Float(0), count: CHUNK_SIZE - chunk.count))
            }
            
            vad.acceptWaveform(samples: chunk)
            samplesBuffer.append(contentsOf: chunk)
            
            let isSpeaking = vad.isSpeechDetected()
            
            if !isRecording && isSpeaking {
                isRecording = true
                recordingStartTime = CACurrentMediaTime()
                let lookbackSize = LOOKBACK_CHUNKS * CHUNK_SIZE
                let lookback = samplesBuffer.suffix(lookbackSize)
                speechBuffer.removeAll()
                speechBuffer.append(contentsOf: lookback)
            }
            
            if isRecording {
                speechBuffer.append(contentsOf: chunk)
                
                if exceedsMaxSpeechDuration() {
                    isRecording = false
                    endRecording()
                    vad.reset()
                }
                else if shouldProcessInterimResults() {
                    processInterimResults()
                    recordingStartTime = CACurrentMediaTime()
                }
            } else {
                trimBufferToLookbackSize()
            }
            
            if isRecording && !isSpeaking {
                isRecording = false
                endRecording()
            }
            
            array = chunkSize < array.count
                ? Array(array[chunkSize..<array.count])
                : []
        }
    }
    
    // MARK: - Processing Helpers
    private func processInterimResults() {
        if speechBuffer.count >= 16000 {
            let result = whisperModel.decode(samples: speechBuffer)
            print(result)
            let transcript = result.text.lowercased()
            if transcript != previousInterim {
                previousInterim = transcript
                processInterim(transcript)
            }
            
        }
    }
    
    private func endRecording() {
        if speechBuffer.count >= 16000 {
            let result = whisperModel.decode(samples: speechBuffer)
            let transcript = result.text.lowercased()
            processFinal(transcript)
            
        }
        
        // Reset state
        speechBuffer.removeAll()
        previousInterim = ""
        recordingStartTime = nil
        lastInterimProcessingTime = nil
    }
    
    private func trimBufferToLookbackSize() {
        let lookbackSize = LOOKBACK_CHUNKS * CHUNK_SIZE
        if samplesBuffer.count > lookbackSize {
            samplesBuffer = Array(samplesBuffer.suffix(lookbackSize))
        }
    }
    
    private func exceedsMaxSpeechDuration() -> Bool {
        return (Double(speechBuffer.count) / Double(SAMPLING_RATE)) > MAX_SPEECH_SECS
    }
    
    private func shouldProcessInterimResults() -> Bool {
        guard let startTime = recordingStartTime else { return false }
        return (CACurrentMediaTime() - startTime) > MIN_REFRESH_SECS &&
               speechBuffer.count >= 16000
    }
    
    // MARK: - Public Control Methods
    func pauseTranscribing() {
        isRecording = false
        speechBuffer.removeAll()
        previousInterim = ""
        recordingStartTime = nil
        lastInterimProcessingTime = nil
        
        // Maintain small overlap for continuity
        let overlapSize = 1000 // About 62.5ms at 16kHz
        if samplesBuffer.count >= overlapSize {
            samplesBuffer = Array(samplesBuffer.suffix(overlapSize))
        } else {
            samplesBuffer.removeAll()
        }
    }
    
    func resumeTranscribing() {
        previousInterim = ""
    }
}

extension AVAudioPCMBuffer {
    func array() -> [Float] {
        guard let ptr = self.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: ptr, count: Int(self.frameLength)))
    }
}
