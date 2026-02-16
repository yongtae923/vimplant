# -*- coding: utf-8 -*-
"""
그리디 스페이싱 결과 분석 스크립트

data/1028_greedy_100610_spacing/ 폴더의 결과를 분석하고 비교합니다.

주요 기능:
1. 모든 결과 데이터 로드 및 정리
2. 스페이싱별/타겟별 통계 분석
3. 비교 시각화 생성
4. 분석 리포트 생성
"""

import os
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 경로 상수
PROJ_ROOT = "C:/Users/user/YongtaeC/vimplant0812"
DATA_BASE = os.path.join(PROJ_ROOT, "data", "1028_greedy_100610_spacing")
OUTPUT_BASE = os.path.join(DATA_BASE, "analysis")

# 스페이싱 목록
SPACING_NAMES = ["0.5mm", "1.0mm", "1.5mm"]

# 시각화 설정
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


def ensure_dir(path: str):
    """디렉토리가 없으면 생성"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def parse_summary(summary_path: str) -> Dict[str, str]:
    """summary.txt 파일을 파싱하여 딕셔너리로 반환"""
    result = {}
    if not os.path.isfile(summary_path):
        return result
    with open(summary_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
    return result


def load_progress_csv(csv_path: str) -> pd.DataFrame:
    """greedy_progress.csv 파일을 로드하여 DataFrame으로 반환"""
    if not os.path.isfile(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_all_results() -> pd.DataFrame:
    """
    모든 결과를 로드하여 통합 DataFrame 생성
    
    Returns:
        DataFrame with columns: spacing, hemisphere, target, selected_count, 
        final_loss, dice, hell, yield, and progress data
    """
    all_data = []
    
    for spacing in SPACING_NAMES:
        spacing_dir = os.path.join(DATA_BASE, spacing)
        if not os.path.isdir(spacing_dir):
            continue
        
        for subdir_name in os.listdir(spacing_dir):
            subdir = os.path.join(spacing_dir, subdir_name)
            if not os.path.isdir(subdir):
                continue
            
            # 폴더명에서 hemisphere와 target 추출
            # 예: "LH_full_spacing_0.5mm" -> "LH", "full"
            match = re.match(r'([LR]H)_(\w+)_spacing_', subdir_name)
            if not match:
                continue
            hemisphere = match.group(1)
            target = match.group(2)
            
            # summary.txt 로드
            summary_path = os.path.join(subdir, "summary.txt")
            summary = parse_summary(summary_path)
            
            # progress CSV 로드
            progress_path = os.path.join(subdir, "greedy_progress.csv")
            progress_df = load_progress_csv(progress_path)
            
            # 최종 메트릭 추출 (progress의 마지막 행)
            final_metrics = {}
            if not progress_df.empty:
                last_row = progress_df.iloc[-1]
                final_metrics = {
                    'final_dice': last_row.get('dice', np.nan),
                    'final_hell': last_row.get('hell', np.nan),
                    'final_yield': last_row.get('yield', np.nan),
                }
            
            # 기본 정보
            row = {
                'spacing': spacing,
                'hemisphere': hemisphere,
                'target': target,
                'selected_count': int(summary.get('selected_count', 0)),
                'final_loss': float(summary.get('final_loss', np.nan)),
                'loss_history_len': int(summary.get('loss_history_len', 0)),
                'subdir': subdir,
                'progress_df': progress_df,
            }
            row.update(final_metrics)
            all_data.append(row)
    
    return pd.DataFrame(all_data)


def compute_statistics(df: pd.DataFrame) -> Dict:
    """통계 분석 수행"""
    stats = {}
    
    # 스페이싱별 통계
    stats['by_spacing'] = {}
    for spacing in SPACING_NAMES:
        spacing_df = df[df['spacing'] == spacing]
        if spacing_df.empty:
            continue
        stats['by_spacing'][spacing] = {
            'count': len(spacing_df),
            'mean_loss': spacing_df['final_loss'].mean(),
            'std_loss': spacing_df['final_loss'].std(),
            'mean_selected_count': spacing_df['selected_count'].mean(),
            'std_selected_count': spacing_df['selected_count'].std(),
            'mean_dice': spacing_df['final_dice'].mean(),
            'std_dice': spacing_df['final_dice'].std(),
            'mean_hell': spacing_df['final_hell'].mean(),
            'std_hell': spacing_df['final_hell'].std(),
            'mean_yield': spacing_df['final_yield'].mean(),
            'std_yield': spacing_df['final_yield'].std(),
        }
    
    # 타겟별 통계
    stats['by_target'] = {}
    for target in df['target'].unique():
        target_df = df[df['target'] == target]
        stats['by_target'][target] = {
            'count': len(target_df),
            'mean_loss': target_df['final_loss'].mean(),
            'mean_dice': target_df['final_dice'].mean(),
            'mean_hell': target_df['final_hell'].mean(),
            'mean_yield': target_df['final_yield'].mean(),
        }
    
    # 반구별 통계
    stats['by_hemisphere'] = {}
    for hem in df['hemisphere'].unique():
        hem_df = df[df['hemisphere'] == hem]
        stats['by_hemisphere'][hem] = {
            'count': len(hem_df),
            'mean_loss': hem_df['final_loss'].mean(),
            'mean_dice': hem_df['final_dice'].mean(),
            'mean_hell': hem_df['final_hell'].mean(),
            'mean_yield': hem_df['final_yield'].mean(),
        }
    
    return stats


def plot_loss_convergence(df: pd.DataFrame, output_dir: str):
    """스페이싱별 loss 수렴 곡선 플롯"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, spacing in enumerate(SPACING_NAMES):
        ax = axes[idx]
        spacing_df = df[df['spacing'] == spacing]
        
        for _, row in spacing_df.iterrows():
            progress_df = row['progress_df']
            if not progress_df.empty and 'loss' in progress_df.columns:
                steps = progress_df['step'].values
                losses = progress_df['loss'].values
                ax.plot(steps, losses, alpha=0.6, linewidth=1.5, 
                       label=f"{row['hemisphere']}_{row['target']}")
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss Convergence - {spacing}')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_convergence_by_spacing.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_comparison(df: pd.DataFrame, output_dir: str):
    """스페이싱별 메트릭 비교 (박스플롯)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['final_loss', 'final_dice', 'final_hell', 'final_yield']
    titles = ['Final Loss', 'Dice Coefficient', 'Hellinger Distance', 'Yield']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        data_for_plot = []
        labels = []
        for spacing in SPACING_NAMES:
            spacing_data = df[df['spacing'] == spacing][metric].dropna()
            if len(spacing_data) > 0:
                data_for_plot.append(spacing_data.values)
                labels.append(spacing)
        
        if data_for_plot:
            bp = ax.boxplot(data_for_plot, tick_labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            ax.set_title(title)
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_comparison_boxplot.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_channel_count_distribution(df: pd.DataFrame, output_dir: str):
    """선택된 채널 수 분포 플롯"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, spacing in enumerate(SPACING_NAMES):
        ax = axes[idx]
        spacing_df = df[df['spacing'] == spacing]
        
        if not spacing_df.empty:
            counts = spacing_df['selected_count'].values
            ax.hist(counts, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(counts.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {counts.mean():.1f}')
            ax.set_xlabel('Selected Channel Count')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Channel Count Distribution - {spacing}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_count_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_spacing_heatmap(df: pd.DataFrame, output_dir: str, metric: str = 'final_loss'):
    """스페이싱 × 타겟 heatmap"""
    # 피벗 테이블 생성
    pivot_data = df.pivot_table(
        values=metric,
        index='target',
        columns='spacing',
        aggfunc='mean'
    )
    
    if pivot_data.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis_r', 
                cbar_kws={'label': metric}, ax=ax)
    ax.set_title(f'{metric.replace("_", " ").title()} by Spacing and Target')
    ax.set_xlabel('Spacing')
    ax.set_ylabel('Target')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'spacing_heatmap_{metric}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_progress_metrics(df: pd.DataFrame, output_dir: str):
    """step별 메트릭 변화 추이"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['dice', 'hell', 'yield', 'loss']
    titles = ['Dice Coefficient', 'Hellinger Distance', 'Yield', 'Loss']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for spacing in SPACING_NAMES:
            spacing_df = df[df['spacing'] == spacing]
            all_steps = []
            all_values = []
            
            for _, row in spacing_df.iterrows():
                progress_df = row['progress_df']
                if not progress_df.empty and metric in progress_df.columns:
                    steps = progress_df['step'].values
                    values = progress_df[metric].values
                    all_steps.extend(steps)
                    all_values.extend(values)
            
            if all_steps:
                # 평균 곡선 계산
                max_step = max(all_steps) if all_steps else 0
                if max_step > 0:
                    step_bins = np.arange(0, max_step + 1, 1)
                    means = []
                    for step in step_bins:
                        step_values = [v for s, v in zip(all_steps, all_values) if s == step]
                        if step_values:
                            means.append(np.mean(step_values))
                        else:
                            means.append(np.nan)
                    ax.plot(step_bins, means, marker='o', label=spacing, 
                           linewidth=2, markersize=4)
        
        ax.set_xlabel('Step')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Progress by Spacing')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'progress_metrics_by_spacing.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_phosphene_map_comparison(df: pd.DataFrame, output_dir: str):
    """동일 타겟에서 스페이싱별 최종 맵 비교"""
    # 타겟별로 그룹화
    for target in df['target'].unique():
        target_df = df[df['target'] == target]
        
        # 반구별로 처리
        for hemisphere in target_df['hemisphere'].unique():
            hem_target_df = target_df[target_df['hemisphere'] == hemisphere]
            
            # 3개 스페이싱 모두 있는지 확인
            available_spacings = hem_target_df['spacing'].unique()
            if len(available_spacings) < 3:
                continue
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'{hemisphere}_{target} - Final Phosphene Maps by Spacing', 
                        fontsize=14, fontweight='bold')
            
            for idx, spacing in enumerate(SPACING_NAMES):
                row = hem_target_df[hem_target_df['spacing'] == spacing]
                if row.empty:
                    continue
                
                subdir = row.iloc[0]['subdir']
                map_path = os.path.join(subdir, 'final_phosphene_map.npy')
                
                if os.path.isfile(map_path):
                    try:
                        phos_map = np.load(map_path)
                        vmax = float(np.max(phos_map)) if np.max(phos_map) > 0 else 1.0
                        axes[idx].imshow(phos_map, cmap='seismic', vmin=0, vmax=vmax)
                        axes[idx].set_title(f'{spacing}\nLoss: {row.iloc[0]["final_loss"]:.4f}')
                        axes[idx].axis('off')
                    except Exception as e:
                        axes[idx].text(0.5, 0.5, f'Error loading\n{str(e)[:30]}', 
                                     ha='center', va='center', transform=axes[idx].transAxes)
                        axes[idx].set_title(spacing)
                else:
                    axes[idx].text(0.5, 0.5, 'File not found', 
                                 ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].set_title(spacing)
                axes[idx].axis('off')
            
            plt.tight_layout()
            safe_target = target.replace('/', '_')
            out_path = os.path.join(output_dir, 'phosphene_maps', 
                                   f'{hemisphere}_{safe_target}_comparison.png')
            ensure_dir(os.path.dirname(out_path))
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()


def generate_comparison_report(df: pd.DataFrame, stats: Dict, output_dir: str):
    """텍스트 리포트 생성"""
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("그리디 스페이싱 결과 분석 리포트\n")
        f.write("=" * 80 + "\n\n")
        
        # 전체 요약
        f.write("1. 전체 요약\n")
        f.write("-" * 80 + "\n")
        f.write(f"총 결과 수: {len(df)}\n")
        f.write(f"스페이싱 종류: {', '.join(SPACING_NAMES)}\n")
        f.write(f"타겟 종류: {', '.join(sorted(df['target'].unique()))}\n")
        f.write(f"반구 종류: {', '.join(sorted(df['hemisphere'].unique()))}\n\n")
        
        # 스페이싱별 통계
        f.write("2. 스페이싱별 통계\n")
        f.write("-" * 80 + "\n")
        for spacing in SPACING_NAMES:
            if spacing in stats['by_spacing']:
                s = stats['by_spacing'][spacing]
                f.write(f"\n[{spacing}]\n")
                f.write(f"  결과 수: {s['count']}\n")
                f.write(f"  평균 Loss: {s['mean_loss']:.6f} ± {s['std_loss']:.6f}\n")
                f.write(f"  평균 선택 채널 수: {s['mean_selected_count']:.1f} ± {s['std_selected_count']:.1f}\n")
                f.write(f"  평균 Dice: {s['mean_dice']:.6f} ± {s['std_dice']:.6f}\n")
                f.write(f"  평균 Hellinger: {s['mean_hell']:.6f} ± {s['std_hell']:.6f}\n")
                f.write(f"  평균 Yield: {s['mean_yield']:.6f} ± {s['std_yield']:.6f}\n")
        f.write("\n")
        
        # 타겟별 통계
        f.write("3. 타겟별 통계\n")
        f.write("-" * 80 + "\n")
        for target in sorted(stats['by_target'].keys()):
            t = stats['by_target'][target]
            f.write(f"\n[{target}]\n")
            f.write(f"  결과 수: {t['count']}\n")
            f.write(f"  평균 Loss: {t['mean_loss']:.6f}\n")
            f.write(f"  평균 Dice: {t['mean_dice']:.6f}\n")
            f.write(f"  평균 Hellinger: {t['mean_hell']:.6f}\n")
            f.write(f"  평균 Yield: {t['mean_yield']:.6f}\n")
        f.write("\n")
        
        # 최고/최저 성능
        f.write("4. 최고/최저 성능\n")
        f.write("-" * 80 + "\n")
        best_loss = df.loc[df['final_loss'].idxmin()]
        worst_loss = df.loc[df['final_loss'].idxmax()]
        f.write(f"\n최저 Loss:\n")
        f.write(f"  {best_loss['spacing']} / {best_loss['hemisphere']} / {best_loss['target']}\n")
        f.write(f"  Loss: {best_loss['final_loss']:.6f}, Dice: {best_loss['final_dice']:.6f}\n")
        f.write(f"\n최고 Loss:\n")
        f.write(f"  {worst_loss['spacing']} / {worst_loss['hemisphere']} / {worst_loss['target']}\n")
        f.write(f"  Loss: {worst_loss['final_loss']:.6f}, Dice: {worst_loss['final_dice']:.6f}\n")
        f.write("\n")
        
        # 타겟별 최적 스페이싱
        f.write("5. 타겟별 최적 스페이싱 (Loss 기준)\n")
        f.write("-" * 80 + "\n")
        for target in sorted(df['target'].unique()):
            target_df = df[df['target'] == target]
            best_spacing = target_df.loc[target_df['final_loss'].idxmin(), 'spacing']
            best_loss_val = target_df['final_loss'].min()
            f.write(f"  {target}: {best_spacing} (Loss: {best_loss_val:.6f})\n")
    
    print(f"[REPORT] Saved: {report_path}")


def export_summary_table(df: pd.DataFrame, stats: Dict, output_dir: str):
    """요약 테이블을 CSV로 저장"""
    # 스페이싱별 요약
    spacing_summary = []
    for spacing in SPACING_NAMES:
        if spacing in stats['by_spacing']:
            s = stats['by_spacing'][spacing]
            spacing_summary.append({
                'spacing': spacing,
                'count': s['count'],
                'mean_loss': s['mean_loss'],
                'std_loss': s['std_loss'],
                'mean_selected_count': s['mean_selected_count'],
                'mean_dice': s['mean_dice'],
                'mean_hell': s['mean_hell'],
                'mean_yield': s['mean_yield'],
            })
    
    spacing_df = pd.DataFrame(spacing_summary)
    spacing_df.to_csv(os.path.join(output_dir, 'summary_by_spacing.csv'), 
                     index=False, encoding='utf-8-sig')
    
    # 전체 데이터 요약
    summary_cols = ['spacing', 'hemisphere', 'target', 'selected_count', 
                    'final_loss', 'final_dice', 'final_hell', 'final_yield']
    df[summary_cols].to_csv(os.path.join(output_dir, 'all_results_summary.csv'), 
                           index=False, encoding='utf-8-sig')
    
    # 스페이싱 비교 테이블
    comparison_data = []
    for target in sorted(df['target'].unique()):
        for hemisphere in sorted(df['hemisphere'].unique()):
            row = {'target': target, 'hemisphere': hemisphere}
            target_hem_df = df[(df['target'] == target) & (df['hemisphere'] == hemisphere)]
            for spacing in SPACING_NAMES:
                spacing_df = target_hem_df[target_hem_df['spacing'] == spacing]
                if not spacing_df.empty:
                    row[f'{spacing}_loss'] = spacing_df.iloc[0]['final_loss']
                    row[f'{spacing}_dice'] = spacing_df.iloc[0]['final_dice']
                    row[f'{spacing}_selected'] = spacing_df.iloc[0]['selected_count']
                else:
                    row[f'{spacing}_loss'] = np.nan
                    row[f'{spacing}_dice'] = np.nan
                    row[f'{spacing}_selected'] = np.nan
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, 'spacing_comparison_table.csv'), 
                         index=False, encoding='utf-8-sig')
    
    print(f"[TABLES] Saved summary tables to {output_dir}")


def main():
    """메인 분석 파이프라인"""
    print("=" * 80)
    print("그리디 스페이싱 결과 분석 시작")
    print("=" * 80)
    
    # 출력 디렉토리 생성
    ensure_dir(OUTPUT_BASE)
    ensure_dir(os.path.join(OUTPUT_BASE, 'figures'))
    ensure_dir(os.path.join(OUTPUT_BASE, 'tables'))
    ensure_dir(os.path.join(OUTPUT_BASE, 'reports'))
    ensure_dir(os.path.join(OUTPUT_BASE, 'phosphene_maps'))
    
    # 데이터 로드
    print("\n[1/6] 데이터 로딩 중...")
    df = load_all_results()
    print(f"  로드된 결과 수: {len(df)}")
    
    if df.empty:
        print("[ERROR] 로드된 데이터가 없습니다.")
        return
    
    # 통계 분석
    print("\n[2/6] 통계 분석 중...")
    stats = compute_statistics(df)
    print(f"  스페이싱별 통계: {len(stats['by_spacing'])}개")
    print(f"  타겟별 통계: {len(stats['by_target'])}개")
    
    # 시각화 생성
    print("\n[3/6] 시각화 생성 중...")
    figures_dir = os.path.join(OUTPUT_BASE, 'figures')
    
    print("  - Loss 수렴 곡선...")
    plot_loss_convergence(df, figures_dir)
    
    print("  - 메트릭 비교 박스플롯...")
    plot_metric_comparison(df, figures_dir)
    
    print("  - 채널 수 분포...")
    plot_channel_count_distribution(df, figures_dir)
    
    print("  - 스페이싱 Heatmap (Loss)...")
    plot_spacing_heatmap(df, figures_dir, 'final_loss')
    
    print("  - 스페이싱 Heatmap (Dice)...")
    plot_spacing_heatmap(df, figures_dir, 'final_dice')
    
    print("  - 진행 메트릭 추이...")
    plot_progress_metrics(df, figures_dir)
    
    print("  - Phosphene 맵 비교...")
    plot_phosphene_map_comparison(df, figures_dir)
    
    # 리포트 생성
    print("\n[4/6] 리포트 생성 중...")
    reports_dir = os.path.join(OUTPUT_BASE, 'reports')
    generate_comparison_report(df, stats, reports_dir)
    
    # 테이블 저장
    print("\n[5/6] 요약 테이블 저장 중...")
    tables_dir = os.path.join(OUTPUT_BASE, 'tables')
    export_summary_table(df, stats, tables_dir)
    
    print("\n[6/6] 완료!")
    print("=" * 80)
    print(f"결과 저장 위치: {OUTPUT_BASE}")
    print("  - figures/: 시각화 이미지")
    print("  - tables/: 요약 테이블 (CSV)")
    print("  - reports/: 분석 리포트 (TXT)")
    print("  - phosphene_maps/: Phosphene 맵 비교 이미지")
    print("=" * 80)


if __name__ == "__main__":
    main()

