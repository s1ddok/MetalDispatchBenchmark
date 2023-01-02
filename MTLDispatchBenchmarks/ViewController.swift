//
//  ViewController.swift
//  MTLDispatchBenchmarks
//
//  Created by Andrey Volodin on 13/06/2019.
//  Copyright © 2019 Andrey Volodin. All rights reserved.
//

import UIKit
import Alloy

typealias BenchmarkResult = (TimeInterval, Configuration)

extension MTLSize: Hashable, CustomDebugStringConvertible {
    public static func == (lhs: MTLSize, rhs: MTLSize) -> Bool {
        return lhs.width == rhs.width && lhs.height == rhs.height && lhs.depth == rhs.depth
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(self.width)
        hasher.combine(self.height)
        hasher.combine(self.depth)
    }
    
    public var debugDescription: String { "\(self.width), \(self.height), \(self.depth)" }
}

struct Configuration: Hashable {
    var threadgroupSize: MTLSize
    var pipelineState: String
}

class ViewController: UIViewController {

    var context: MTLContext!

    var exactState: MTLComputePipelineState!
    var uniformState: MTLComputePipelineState!
    var optimizedUniformState: MTLComputePipelineState!

    var texture1: MTLTexture!
    var texture2: MTLTexture!

    override func viewDidLoad() {
        super.viewDidLoad()

        let context = try! MTLContext(device: Metal.device)

        self.context = context
        
        guard
            let library = try? self.context.library(for: ViewController.self),
            let exactPipelineState = try? library.computePipelineState(function: "exactIncrement"),
            let uniformPipelineState = try? library.computePipelineState(function: "uniformIncrement")
        else {
            fatalError()
        }

        let kernelDescriptor = MTLComputePipelineDescriptor()
        kernelDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true

        let function = library.makeFunction(name: "exactIncrement")

        kernelDescriptor.computeFunction = function

        guard let uniformFusedPipelineState = try? context.device
                                                          .makeComputePipelineState(descriptor: kernelDescriptor,
                                                                                    options: [],
                                                                                    reflection: nil)
        else {
            fatalError()
        }


        let texture1 = try! context.texture(width: 513, height: 513, pixelFormat: .rgba32Sint, usage: [.shaderRead, .shaderWrite])
        let texture2 = try! context.texture(width: 513, height: 513, pixelFormat: .rgba32Sint, usage: [.shaderRead, .shaderWrite])

        self.texture1 = texture1
        self.texture2 = texture2

        self.exactState = exactPipelineState
        self.uniformState = uniformPipelineState
        self.optimizedUniformState = uniformFusedPipelineState

        print(self.exactState.threadExecutionWidth)
        print(self.uniformState.threadExecutionWidth)
        print(self.optimizedUniformState.threadExecutionWidth)
    }


    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(true)
        
        // warm-up run
        _ = self.run()
        
        let results = self.run()
        
        let normalizedResults = results.map { ($0.0 / results.first!.0, $0.1.pipelineState, $0.1.threadgroupSize) }
        normalizedResults.forEach { print($0) }
        
        let groupedByThreadgroup = Dictionary(grouping: results) { $0.1.threadgroupSize }
    
        for (key, results) in groupedByThreadgroup {
            print("Best results for key: \(key)")
            print(results.sorted { $0.0 < $1.0 }.map { ($0.0, $0.1.pipelineState) }[0..<3])
        }
        
        let groupedByDispatch = Dictionary(grouping: results) { $0.1.pipelineState}
        for (key, results) in groupedByDispatch {
            print("Average results for key: \(key)")
            print(results.reduce(0.0, { $0 + $1.0 }) / TimeInterval(results.count))
        }
    }
    
    func run(iterations: Int = 400, repeatIterations: Int = 20) -> [BenchmarkResult] {
        var results: [(TimeInterval, Configuration)] = []
        
        for _ in 0..<repeatIterations {
            let possibleDivisors = (0...5).map { Int(pow(2.0, Double($0))) }.shuffled()
            
            for d in possibleDivisors {
                var dispatchGroupSize = self.uniformState.max2dThreadgroupSize
                dispatchGroupSize.height /= d
                
                var calls = [() -> [BenchmarkResult]]()
                calls.append( {
                    self.testDispatch(configuration: Configuration(threadgroupSize: dispatchGroupSize,
                                                                   pipelineState: "uniform"),
                                      iterations: iterations)
                })
                
                calls.append({
                    self.testDispatch(configuration: Configuration(threadgroupSize: dispatchGroupSize,
                                                                   pipelineState: "uniformOptimized"),
                                      iterations: iterations)
                })
                    
                calls.append({
                    self.testDispatch(configuration: Configuration(threadgroupSize: dispatchGroupSize,
                                                                   pipelineState: "exact"),
                                      iterations: iterations)
                })
                
                calls.lazy.shuffled().map { $0() }.forEach { results.append(contentsOf: $0) }
            }
        }
        
        return results.sorted { $0.0 < $1.0 }
    }

    func testDispatch(configuration: Configuration, iterations: Int) -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        try? self.context.scheduleAndWait { buffer in
            buffer.compute { encoder in
                for i in 1...iterations {
                    let t1 = i % 2 == 0 ? self.texture1 : self.texture2
                    let t2 = i % 2 == 0 ? self.texture2 : self.texture1

                    encoder.setTextures(t1, t2)

                    let state: MTLComputePipelineState

                    switch configuration.pipelineState {
                    case "uniform": state = self.uniformState
                    case "uniformOptimized": state = self.optimizedUniformState
                    case "exact": state = self.exactState
                    default: fatalError()
                    }

                    if configuration.pipelineState != "exact" {
                        encoder.dispatch2d(state: state,
                                           covering: t1!.size,
                                           threadgroupSize: configuration.threadgroupSize)
                    } else {
                        encoder.dispatch2d(state: state,
                                           exactly: t1!.size,
                                           threadgroupSize: configuration.threadgroupSize)
                    }
                }

                buffer.addCompletedHandler { _buffer in
                    results.append((_buffer.gpuExecutionTime, configuration))
                }
            }
        }
        
        return results
    }


}

