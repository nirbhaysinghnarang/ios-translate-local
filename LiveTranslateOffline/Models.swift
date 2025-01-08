//
//  Models.swift
//  LiveTranslateOffline
//
//  Created by Nirbhay Singh Narang on 1/8/25.
//

import Foundation
func getWhisperModel(sourceLanguage: String) -> SherpaOnnxOfflineRecognizer {
    let whisperConfig = sherpaOnnxOfflineWhisperModelConfig(
        encoder: Bundle.main.path(forResource: "base-encoder.int8", ofType: "onnx") ?? "",
        decoder: Bundle.main.path(forResource: "base-decoder.int8", ofType: "onnx") ?? "",
        language: sourceLanguage.lowercased(),
        task: "transcribe",
        tailPaddings: 10000
    )
    let featConfig = sherpaOnnxFeatureConfig(
        sampleRate: 16000,
        featureDim: 80
    )
    let modelConfig = sherpaOnnxOfflineModelConfig(
        tokens: Bundle.main.path(forResource: "base-tokens", ofType: "txt") ?? "",
        whisper: whisperConfig,
        numThreads: 2,
        provider: "cpu"
    )
    var config = sherpaOnnxOfflineRecognizerConfig(
        featConfig: featConfig,
        modelConfig: modelConfig,
        decodingMethod: "greedy_search",
        maxActivePaths: 4
    )
    
    return SherpaOnnxOfflineRecognizer(config: &config)
}
