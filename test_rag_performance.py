#!/usr/bin/env python3
"""
RAG性能測試腳本
用於測試優化後的RAG系統查詢速度和準確性
"""

import time
import json
import sys
import os
from datetime import datetime

# 添加rag_system到路徑
sys.path.append('rag_system')
from tcm_rag import TCMRAGSystem

def test_rag_performance():
    """測試RAG系統性能"""
    
    print("=" * 60)
    print("RAG系統性能測試")
    print("=" * 60)
    
    # 初始化RAG系統
    print("初始化RAG系統...")
    rag_system = TCMRAGSystem()
    
    # 加載系統
    if not rag_system.load_system():
        print("❌ RAG系統加載失敗")
        return
    
    print("✅ RAG系統加載成功")
    
    # 測試問題列表
    test_questions = [
        "什麼是氣血兩虛？",
        "脾胃虛弱的症狀有哪些？",
        "六君子湯的組成是什麼？",
        "如何調理陰虛火旺？",
        "小柴胡湯適用於什麼證候？"
    ]
    
    results = []
    total_start_time = time.time()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 測試問題 {i}: {question}")
        print("-" * 40)
        
        # 測試查詢時間
        start_time = time.time()
        result = rag_system.generate_answer(question, use_rag=True)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # 顯示結果
        print(f"⏱️  響應時間: {response_time:.2f}秒")
        print(f"📖 檢索到知識: {result['retrieved_knowledge']} 個片段")
        print(f"📝 上下文長度: {result['context_length']} 字符")
        print(f"💬 回答: {result['answer'][:100]}..." if len(result['answer']) > 100 else f"💬 回答: {result['answer']}")
        
        if result['knowledge_sources']:
            print("📚 知識來源:")
            for source in result['knowledge_sources']:
                print(f"   • {source['book']} / {source['chapter']}")
        
        # 記錄結果
        results.append({
            "question": question,
            "response_time": response_time,
            "retrieved_knowledge": result['retrieved_knowledge'],
            "context_length": result['context_length'],
            "answer_length": len(result['answer']),
            "knowledge_sources": len(result['knowledge_sources'])
        })
    
    total_time = time.time() - total_start_time
    avg_time = total_time / len(test_questions)
    
    # 統計報告
    print("\n" + "=" * 60)
    print("📊 性能統計報告")
    print("=" * 60)
    print(f"總測試時間: {total_time:.2f}秒")
    print(f"平均查詢時間: {avg_time:.2f}秒")
    print(f"最快查詢: {min(r['response_time'] for r in results):.2f}秒")
    print(f"最慢查詢: {max(r['response_time'] for r in results):.2f}秒")
    
    response_times = [r['response_time'] for r in results]
    fast_queries = len([t for t in response_times if t < 10])
    medium_queries = len([t for t in response_times if 10 <= t < 20])
    slow_queries = len([t for t in response_times if t >= 20])
    
    print(f"\n⚡ 快速查詢 (<10秒): {fast_queries}")
    print(f"🔄 中等查詢 (10-20秒): {medium_queries}")  
    print(f"🐌 慢速查詢 (≥20秒): {slow_queries}")
    
    # 保存詳細結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"rag_performance_test_{timestamp}.json"
    
    test_summary = {
        "test_time": datetime.now().isoformat(),
        "total_questions": len(test_questions),
        "total_time": total_time,
        "average_time": avg_time,
        "min_time": min(response_times),
        "max_time": max(response_times),
        "performance_distribution": {
            "fast": fast_queries,
            "medium": medium_queries, 
            "slow": slow_queries
        },
        "detailed_results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 詳細結果已保存到: {output_file}")
    
    # 性能評估
    print("\n" + "=" * 60)
    print("📈 性能評估")
    print("=" * 60)
    
    if avg_time < 5:
        print("🎉 優秀：平均響應時間 < 5秒")
    elif avg_time < 10:
        print("👍 良好：平均響應時間 5-10秒")
    elif avg_time < 20:
        print("⚠️  一般：平均響應時間 10-20秒")
    else:
        print("❌ 需要優化：平均響應時間 > 20秒")
    
    if fast_queries >= len(test_questions) * 0.8:
        print("🚀 查詢速度分布良好：80%以上查詢 < 10秒")
    elif fast_queries >= len(test_questions) * 0.6:
        print("✅ 查詢速度分布可接受：60%以上查詢 < 10秒")
    else:
        print("⚠️  查詢速度分布需改善：< 60%查詢 < 10秒")

def test_cache_effectiveness():
    """測試快取效果"""
    print("\n" + "=" * 60)
    print("🎯 快取效果測試")
    print("=" * 60)
    
    rag_system = TCMRAGSystem()
    if not rag_system.load_system():
        print("❌ RAG系統加載失敗")
        return
    
    test_question = "什麼是氣血兩虛？"
    
    print(f"測試問題: {test_question}")
    
    # 第一次查詢（無快取）
    print("\n🔄 第一次查詢（無快取）...")
    start_time = time.time()
    result1 = rag_system.generate_answer(test_question, use_rag=True)
    time1 = time.time() - start_time
    
    print(f"⏱️  第一次查詢時間: {time1:.2f}秒")
    
    # 第二次查詢（有快取）
    print("\n⚡ 第二次查詢（快取命中）...")
    start_time = time.time()
    result2 = rag_system.generate_answer(test_question, use_rag=True)
    time2 = time.time() - start_time
    
    print(f"⏱️  第二次查詢時間: {time2:.2f}秒")
    
    # 快取效果
    if time2 < time1:
        improvement = ((time1 - time2) / time1) * 100
        print(f"\n🎉 快取效果: 提升 {improvement:.1f}% (節省 {time1 - time2:.2f}秒)")
    else:
        print(f"\n⚠️  快取未生效或其他因素影響")

if __name__ == "__main__":
    try:
        test_rag_performance()
        test_cache_effectiveness()
    except KeyboardInterrupt:
        print("\n測試被用戶中斷")
    except Exception as e:
        print(f"\n測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()