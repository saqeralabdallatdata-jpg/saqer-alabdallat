import numpy as np
from typing import Dict, List, Any

class StatisticalProcessControlEngine:
    @staticmethod
    def calculate_ewma(data_points: List[float], lambda_param: float = 0.2) -> List[float]:
        """Computes Exponentially Weighted Moving Average (EWMA) for chemical or physical dimension shifts"""
        if not data_points:
            return []
        ewma_values = [data_points[0]]
        for i in range(1, len(data_points)):
            next_val = lambda_param * data_points[i] + (1 - lambda_param) * ewma_values[-1]
            ewma_values.append(float(next_val))
        return ewma_values

    @staticmethod
    def execute_pareto_analysis(defect_logs: Dict[str, int]) -> List[Dict[str, Any]]:
        """Calculates Pareto Analysis to identify the 20% root causes driving 80% of factory defects"""
        total_defects = sum(defect_logs.values())
        if total_defects == 0:
            return []
            
        # الترتيب التنازلي للعيوب
        sorted_defects = sorted(defect_logs.items(), key=lambda x: x[1], reverse=True)
        
        cumulative_sum = 0
        pareto_results = []
        
        for defect, count in sorted_defects:
            cumulative_sum += count
            cumulative_pct = (cumulative_sum / total_defects) * 100
            pareto_results.append({
                "defect_type": defect,
                "count": count,
                "percentage": round((count / total_defects) * 100, 2),
                "cumulative_percentage": round(cumulative_pct, 2),
                "rank_vital_few": cumulative_pct <= 80.0
            })
            
        return pareto_results