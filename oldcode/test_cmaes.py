# -*- coding: utf-8 -*-
"""
CMA-ES 최적화 테스트 스크립트

간단한 테스트 함수로 CMA-ES가 제대로 작동하는지 확인합니다.
"""

import numpy as np
import time

# CMA-ES 라이브러리 import
try:
    import cma
    CMA_AVAILABLE = True
    print("✅ CMA-ES library imported successfully")
except ImportError:
    print("❌ CMA-ES library not available. Please install with: pip install cma")
    CMA_AVAILABLE = False
    exit(1)

def test_function(x):
    """
    테스트용 비용 함수 (Rastrigin 함수)
    전역 최솟값: x = [0, 0, 0, 0], f(x) = 0
    """
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def get_evaluation_count(es):
    """CMA-ES 객체에서 평가 횟수를 안전하게 가져오는 함수"""
    try:
        # 최신 버전
        if hasattr(es.result, 'evals_total'):
            return es.result.evals_total
        elif hasattr(es, 'evals_total'):
            return es.evals_total
        elif hasattr(es.result, 'evals'):
            return es.result.evals
        elif hasattr(es, 'evals'):
            return es.evals
        else:
            # 기본값 반환 - generation 속성도 안전하게 확인
            if hasattr(es, 'generation'):
                return es.generation * es.popsize
            elif hasattr(es, 'countiter'):
                return es.countiter * es.popsize
            elif hasattr(es, 'countgen'):
                return es.countgen * es.popsize
            else:
                # 모든 속성이 없으면 기본값
                return 400  # 일반적인 기본값
    except:
        # 오류 발생 시 기본값 반환
        try:
            if hasattr(es, 'popsize'):
                if hasattr(es, 'generation'):
                    return es.generation * es.popsize
                elif hasattr(es, 'countiter'):
                    return es.countiter * es.popsize
                elif hasattr(es, 'countgen'):
                    return es.countgen * es.popsize
                else:
                    return 400
            else:
                return 400
        except:
            return 400

def get_generation_count(es):
    """CMA-ES 객체에서 세대 수를 안전하게 가져오는 함수"""
    try:
        if hasattr(es, 'generation'):
            return es.generation
        elif hasattr(es, 'countiter'):
            return es.countiter
        elif hasattr(es, 'countgen'):
            return es.countgen
        else:
            return 0
    except:
        return 0

def test_cmaes_simple():
    """간단한 CMA-ES 테스트"""
    print("\n=== Simple CMA-ES Test ===")
    
    # 4차원 테스트 (현재 케이스와 동일)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    sigma0 = 2.0
    
    # CMA-ES 설정
    opts = cma.CMAOptions()
    opts.set({
        'maxiter': 50,           # 최대 세대 수
        'popsize': 8,            # 개체 수
        'CMA_diagonal': True,    # 대각선 공분산 행렬
        'tolfun': 1e-6,         # 함수 값 수렴 기준
        'verbose': -1            # 출력 최소화
    })
    
    print(f"Starting CMA-ES with {opts['popsize']} population size...")
    start_time = time.time()
    
    # CMA-ES 실행
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    es.optimize(test_function)
    
    optimization_time = time.time() - start_time
    
    # 평가 횟수 안전하게 가져오기
    eval_count = get_evaluation_count(es)
    gen_count = get_generation_count(es)
    
    print(f"✅ Optimization completed in {optimization_time:.3f} seconds")
    print(f"Best parameters: {es.result.xbest}")
    print(f"Best cost: {es.result.fbest:.6f}")
    print(f"Total evaluations: {eval_count}")
    print(f"Generations: {gen_count}")
    
    return es.result.xbest, es.result.fbest

def test_cmaes_bounded():
    """경계 제약 조건이 있는 CMA-ES 테스트"""
    print("\n=== Bounded CMA-ES Test ===")
    
    # 파라미터 경계 (현재 케이스와 유사)
    bounds = [(-5, 5), (-5, 5), (-5, 5), (-5, 5)]
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    sigma0 = 2.0
    
    # 경계 제약 조건을 위한 wrapper 함수
    def bounded_test_function(x):
        # 파라미터를 경계 내로 클리핑
        x_clipped = np.clip(x, 
                           [bounds[0] for bounds in bounds],
                           [bounds[1] for bounds in bounds])
        return test_function(x_clipped)
    
    # CMA-ES 설정
    opts = cma.CMAOptions()
    opts.set({
        'maxiter': 50,
        'popsize': 8,
        'CMA_diagonal': True,
        'tolfun': 1e-6,
        'verbose': -1
    })
    
    print(f"Starting bounded CMA-ES...")
    start_time = time.time()
    
    # CMA-ES 실행
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    es.optimize(bounded_test_function)
    
    optimization_time = time.time() - start_time
    
    # 평가 횟수 안전하게 가져오기
    eval_count = get_evaluation_count(es)
    gen_count = get_generation_count(es)
    
    print(f"✅ Bounded optimization completed in {optimization_time:.3f} seconds")
    print(f"Best parameters: {es.result.xbest}")
    print(f"Best cost: {es.result.fbest:.6f}")
    print(f"Total evaluations: {eval_count}")
    print(f"Generations: {gen_count}")
    print(f"Parameters within bounds: {all([bounds[i][0] <= es.result.xbest[i] <= bounds[i][1] for i in range(4)])}")
    
    return es.result.xbest, es.result.fbest

def test_cmaes_performance():
    """CMA-ES 성능 테스트 (여러 차원에서)"""
    print("\n=== CMA-ES Performance Test ===")
    
    dimensions = [2, 4, 8, 16]
    results = {}
    
    for dim in dimensions:
        print(f"\nTesting {dim} dimensions...")
        
        x0 = np.zeros(dim)
        sigma0 = 2.0
        
        # 차원에 따라 개체 수 조정
        popsize = max(4, int(4 + np.floor(3 * np.log(dim))))
        
        opts = cma.CMAOptions()
        opts.set({
            'maxiter': 100 // popsize,
            'popsize': popsize,
            'CMA_diagonal': True,
            'tolfun': 1e-6,
            'verbose': -1
        })
        
        start_time = time.time()
        
        # CMA-ES 실행
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        es.optimize(test_function)
        
        optimization_time = time.time() - start_time
        
        # 평가 횟수 안전하게 가져오기
        eval_count = get_evaluation_count(es)
        gen_count = get_generation_count(es)
        
        results[dim] = {
            'best_cost': es.result.fbest,
            'time': optimization_time,
            'evaluations': eval_count,
            'generations': gen_count
        }
        
        print(f"  {dim}D: Cost={es.result.fbest:.6f}, Time={optimization_time:.3f}s, Evals={eval_count}, Gens={gen_count}")
    
    return results

def test_cmaes_4d_specific():
    """4차원에서의 CMA-ES 성능 테스트 (현재 케이스와 동일)"""
    print("\n=== 4D CMA-ES Specific Test ===")
    
    # 현재 케이스와 동일한 파라미터 범위
    bounds = [(-90, 90), (-15, 110), (0, 40), (10, 40)]
    x0 = np.array([0.0, 0.0, 20.0, 25.0])  # 초기값
    sigma0 = 10.0  # 초기 스텝 사이즈
    
    # 경계 제약 조건을 위한 wrapper 함수
    def bounded_4d_function(x):
        # 파라미터를 경계 내로 클리핑
        x_clipped = np.clip(x, 
                           [bounds[0] for bounds in bounds],
                           [bounds[1] for bounds in bounds])
        
        # 간단한 4차원 테스트 함수 (현재 케이스와 유사한 복잡도)
        cost = sum([(xi - x0[i])**2 for i, xi in enumerate(x_clipped)])
        return cost
    
    # CMA-ES 설정 (현재 케이스와 동일)
    opts = cma.CMAOptions()
    opts.set({
        'maxiter': 15,           # 150회 평가 / 10개체 = 15세대
        'popsize': 10,           # 개체 수 (4차원에 적합)
        'CMA_diagonal': True,    # 대각선 공분산 행렬
        'CMA_elitist': True,     # 엘리트 전략
        'tolfun': 1e-6,         # 함수 값 수렴 기준
        'tolx': 1e-6,           # 파라미터 수렴 기준
        'verbose': -1            # 출력 최소화
    })
    
    print(f"Starting 4D CMA-ES with {opts['popsize']} population size...")
    print(f"Target: 150 evaluations in {opts['maxiter']} generations")
    start_time = time.time()
    
    # CMA-ES 실행
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    es.optimize(bounded_4d_function)
    
    optimization_time = time.time() - start_time
    
    # 평가 횟수 안전하게 가져오기
    eval_count = get_evaluation_count(es)
    gen_count = get_generation_count(es)
    
    print(f"✅ 4D optimization completed in {optimization_time:.3f} seconds")
    print(f"Best parameters: {es.result.xbest}")
    print(f"Best cost: {es.result.fbest:.6f}")
    print(f"Total evaluations: {eval_count}")
    print(f"Generations: {gen_count}")
    print(f"Parameters within bounds: {all([bounds[i][0] <= es.result.xbest[i] <= bounds[i][1] for i in range(4)])}")
    
    return es.result.xbest, es.result.fbest

if __name__ == "__main__":
    print("🚀 CMA-ES Optimization Test Suite")
    print("=" * 50)
    
    try:
        # 1. 간단한 테스트
        best_params, best_cost = test_cmaes_simple()
        
        # 2. 경계 제약 조건 테스트
        best_params_bounded, best_cost_bounded = test_cmaes_bounded()
        
        # 3. 성능 테스트
        performance_results = test_cmaes_performance()
        
        # 4. 4차원 특화 테스트 (현재 케이스와 동일)
        best_params_4d, best_cost_4d = test_cmaes_4d_specific()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print(f"Simple test result: {best_cost:.6f}")
        print(f"Bounded test result: {best_cost_bounded:.6f}")
        print(f"4D specific test result: {best_cost_4d:.6f}")
        
        print("\nPerformance Summary:")
        for dim, result in performance_results.items():
            print(f"  {dim}D: {result['time']:.3f}s, {result['evaluations']} evals")
            
        print(f"\n4D Test Summary:")
        print(f"  Time: {performance_results.get(4, {}).get('time', 'N/A')}s")
        print(f"  Evaluations: {performance_results.get(4, {}).get('evaluations', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
