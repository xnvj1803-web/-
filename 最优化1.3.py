# ==================== 1. 数据加载与预处理 ====================
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import KFold
import os
import gc
import warnings
warnings.filterwarnings('ignore')

def load_ratings_only(ratings_path=r"E:\大三上\最优化\大作业\ml-20m\ratings.csv", 
                      sample_size=50000, sample_frac=None):
    """
    只读取ratings.csv文件
    """
    print("正在加载ratings.csv...")
    
    if sample_size is not None:
        print(f"读取前 {sample_size:,} 行数据...")
        ratings = pd.read_csv(ratings_path, nrows=sample_size)
    elif sample_frac is not None:
        print(f"采样 {sample_frac*100:.1f}% 的数据...")
        ratings = pd.read_csv(ratings_path)
        ratings = ratings.sample(frac=sample_frac, random_state=42)
    else:
        ratings = pd.read_csv(ratings_path)
    
    print(f"数据集信息:")
    print(f"  总评分数量: {len(ratings):,}")
    print(f"  用户数量: {ratings['userId'].nunique():,}")
    print(f"  电影数量: {ratings['movieId'].nunique():,}")
    print(f"  评分范围: {ratings['rating'].min()} - {ratings['rating'].max()}")
    
    # 创建映射
    users = sorted(ratings['userId'].unique())
    movies = sorted(ratings['movieId'].unique())
    
    user_id_map = {user_id: i for i, user_id in enumerate(users)}
    movie_id_map = {movie_id: i for i, movie_id in enumerate(movies)}
    
    ratings['user_idx'] = ratings['userId'].map(user_id_map)
    ratings['movie_idx'] = ratings['movieId'].map(movie_id_map)
    
    # 创建稀疏矩阵
    n_users = len(users)
    n_movies = len(movies)
    
    print(f"创建评分矩阵: {n_users} × {n_movies}")
    print(f"矩阵大小: {n_users * n_movies:,} 个元素")
    print(f"稀疏度: {(1 - len(ratings) / (n_users * n_movies)) * 100:.2f}%")
    
    ratings_matrix = sparse.csr_matrix(
        (ratings['rating'].astype(np.float32), 
         (ratings['user_idx'], ratings['movie_idx'])),
        shape=(n_users, n_movies),
        dtype=np.float32
    )
    
    del users, movies, user_id_map, movie_id_map
    gc.collect()
    
    return ratings_matrix, ratings, n_users, n_movies

def create_cv_split(ratings_df, n_folds=5, random_state=42):
    """创建交叉验证划分"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return kf.split(np.arange(len(ratings_df)))

def create_train_test_indices(ratings_df, train_idx, test_idx):
    """创建训练和测试集的索引和评分"""
    train_ratings = ratings_df.iloc[train_idx]
    test_ratings = ratings_df.iloc[test_idx]
    
    train_data = {
        'users': train_ratings['user_idx'].values,
        'movies': train_ratings['movie_idx'].values,
        'ratings': train_ratings['rating'].values.astype(np.float32)
    }
    
    test_data = {
        'users': test_ratings['user_idx'].values,
        'movies': test_ratings['movie_idx'].values,
        'ratings': test_ratings['rating'].values.astype(np.float32)
    }
    
    return train_data, test_data

# ==================== 2. 修正的凸方法实现 ====================
def nuclear_norm_svt_corrected(observed_matrix, mask, lambda_reg=1.0, max_iter=50, 
                              epsilon=1e-6):
    """
    修正的核范数最小化
    关键修正：正确处理观测矩阵和掩码
    """
    m, n = observed_matrix.shape
    
    # 初始化：使用观测值的均值填充
    observed_mean = np.sum(observed_matrix) / np.sum(mask)
    X = np.full((m, n), observed_mean, dtype=np.float32)
    Y = X.copy()
    
    for k in range(max_iter):
        # 对Y进行SVD
        try:
            U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        except np.linalg.LinAlgError:
            # 尝试随机SVD
            try:
                from sklearn.utils.extmath import randomized_svd
                U, S, Vt = randomized_svd(Y, n_components=min(10, min(m, n)-1), 
                                         random_state=42)
            except:
                break
        
        # 软阈值操作
        S_thresh = np.maximum(S - lambda_reg, 0)
        
        # 重建矩阵
        X_new = U @ np.diag(S_thresh) @ Vt
        
        # 更新Y：只更新观测到的位置
        Y = X_new + mask * (observed_matrix - X_new)
        
        # 检查收敛
        diff = np.linalg.norm(X_new - X, 'fro') / (np.linalg.norm(X, 'fro') + 1e-10)
        if diff < epsilon:
            print(f"凸方法在迭代 {k+1} 收敛 (相对变化: {diff:.6f})")
            break
        
        X = X_new
        
        if (k + 1) % 10 == 0:
            print(f"凸方法迭代 {k+1}/{max_iter}, 相对变化: {diff:.6f}")
    
    # 确保预测值在合理范围内
    X = np.clip(X, 0.5, 5.0)
    return X

def soft_impute_corrected(observed_matrix, mask, lambda_reg=0.5, max_iter=50, 
                         epsilon=1e-6):
    """
    修正的SoftImpute算法
    """
    m, n = observed_matrix.shape
    
    # 初始化：用观测值的均值填充
    observed_mean = np.sum(observed_matrix) / np.sum(mask)
    X = np.full((m, n), observed_mean, dtype=np.float32)
    
    for k in range(max_iter):
        # 用当前估计填充缺失值
        Z = mask * observed_matrix + (1 - mask) * X
        
        # SVD分解
        try:
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            try:
                from sklearn.utils.extmath import randomized_svd
                U, S, Vt = randomized_svd(Z, n_components=min(10, min(m, n)-1), 
                                         random_state=42)
            except:
                break
        
        # 软阈值
        S_thresh = np.maximum(S - lambda_reg, 0)
        
        # 更新X
        X_new = U @ np.diag(S_thresh) @ Vt
        
        # 检查收敛
        diff = np.linalg.norm(X_new - X, 'fro') / (np.linalg.norm(X, 'fro') + 1e-10)
        if diff < epsilon:
            print(f"SoftImpute在迭代 {k+1} 收敛 (相对变化: {diff:.6f})")
            break
        
        X = X_new
        
        if (k + 1) % 10 == 0:
            print(f"SoftImpute迭代 {k+1}/{max_iter}, 相对变化: {diff:.6f}")
    
    # 确保预测值在合理范围内
    X = np.clip(X, 0.5, 5.0)
    return X

# ==================== 3. 修正的非凸方法实现 ====================
class CorrectedALSMatrixCompletion:
    """修正的交替最小二乘法，解决过拟合问题"""
    
    def __init__(self, n_factors=10, reg=0.5, max_iter=10, global_mean=None):
        self.n_factors = n_factors
        self.reg = reg  # 增加正则化
        self.max_iter = max_iter
        self.global_mean = global_mean
        
    def fit(self, train_data, n_users, n_movies):
        self.n_users = n_users
        self.n_movies = n_movies
        
        # 初始化
        self.U = np.random.randn(n_users, self.n_factors).astype(np.float32) * 0.1
        self.V = np.random.randn(n_movies, self.n_factors).astype(np.float32) * 0.1
        
        users = train_data['users']
        movies = train_data['movies']
        ratings = train_data['ratings']
        
        # 使用传入的全局均值或计算
        if self.global_mean is None:
            self.global_mean = np.mean(ratings)
        
        # 中心化评分
        ratings_centered = ratings - self.global_mean
        
        # 创建用户和电影的评分映射
        user_to_ratings = {}
        movie_to_ratings = {}
        
        for i in range(len(ratings)):
            u = users[i]
            m = movies[i]
            r = ratings_centered[i]
            
            if u not in user_to_ratings:
                user_to_ratings[u] = {'movies': [], 'ratings': []}
            if m not in movie_to_ratings:
                movie_to_ratings[m] = {'users': [], 'ratings': []}
                
            user_to_ratings[u]['movies'].append(m)
            user_to_ratings[u]['ratings'].append(r)
            
            movie_to_ratings[m]['users'].append(u)
            movie_to_ratings[m]['ratings'].append(r)
        
        for iteration in range(self.max_iter):
            # 更新用户因子
            for u in range(n_users):
                if u in user_to_ratings:
                    movies_u = user_to_ratings[u]['movies']
                    ratings_u = user_to_ratings[u]['ratings']
                    
                    V_u = self.V[movies_u]
                    A = V_u.T @ V_u + self.reg * np.eye(self.n_factors, dtype=np.float32)
                    b = V_u.T @ ratings_u
                    
                    try:
                        self.U[u] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        self.U[u] = np.linalg.lstsq(A, b, rcond=1e-3)[0]
            
            # 更新电影因子
            for m in range(n_movies):
                if m in movie_to_ratings:
                    users_m = movie_to_ratings[m]['users']
                    ratings_m = movie_to_ratings[m]['ratings']
                    
                    U_m = self.U[users_m]
                    A = U_m.T @ U_m + self.reg * np.eye(self.n_factors, dtype=np.float32)
                    b = U_m.T @ ratings_m
                    
                    try:
                        self.V[m] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        self.V[m] = np.linalg.lstsq(A, b, rcond=1e-3)[0]
            
            # 计算训练误差
            if (iteration + 1) % 2 == 0:
                train_pred = self.predict(users, movies)
                train_rmse = np.sqrt(np.mean((ratings - train_pred) ** 2))
                print(f"  迭代 {iteration+1}/{self.max_iter}: 训练RMSE={train_rmse:.4f}")
        
        return self
    
    def predict(self, users, movies):
        predictions = np.zeros(len(users), dtype=np.float32)
        for i, (u, m) in enumerate(zip(users, movies)):
            predictions[i] = np.dot(self.U[u], self.V[m]) + self.global_mean
        
        predictions = np.clip(predictions, 0.5, 5.0)
        return predictions

# ==================== 4. 评估函数 ====================
def calculate_rmse(predictions, true_ratings):
    if len(predictions) == 0:
        return float('inf')
    mse = np.mean((predictions - true_ratings) ** 2)
    return np.sqrt(mse)

# ==================== 5. 参数调优的实验函数 ====================
def run_parameter_tuning_experiment():
    """运行参数调优实验"""
    print("=" * 60)
    print("MovieLens 20M 参数调优实验")
    print("=" * 60)
    
    print("\n1. 加载数据...")
    ratings_matrix, ratings_df, n_users, n_movies = load_ratings_only(
        sample_size=50000
    )
    
    print(f"\n2. 数据统计:")
    print(f"   用户数: {n_users}")
    print(f"   电影数: {n_movies}")
    print(f"   评分数: {len(ratings_df)}")
    
    print("\n3. 准备3折交叉验证...")
    cv_splits = create_cv_split(ratings_df, n_folds=3)
    
    # 测试不同的正则化参数
    lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    best_results = {
        'convex_svt': {'lambda': None, 'rmse': float('inf'), 'scores': []},
        'convex_softimpute': {'lambda': None, 'rmse': float('inf'), 'scores': []},
        'als': {'lambda': None, 'rmse': float('inf'), 'scores': []}
    }
    
    fold_idx = 1
    for train_idx, test_idx in cv_splits:
        print(f"\n{'='*40}")
        print(f"折 {fold_idx}/3")
        print(f"{'='*40}")
        
        train_data, test_data = create_train_test_indices(ratings_df, train_idx, test_idx)
        
        print(f"  训练集大小: {len(train_data['users']):,}")
        print(f"  测试集大小: {len(test_data['users']):,}")
        
        # 基线方法
        global_mean = np.mean(train_data['ratings'])
        baseline_predictions = np.full(len(test_data['ratings']), global_mean)
        baseline_rmse = calculate_rmse(baseline_predictions, test_data['ratings'])
        print(f"\n  基线RMSE: {baseline_rmse:.4f}")
        
        # 测试不同的正则化参数
        for lambda_reg in lambda_values:
            print(f"\n  --- 正则化参数 λ = {lambda_reg} ---")
            
            # 创建训练矩阵
            train_matrix = np.zeros((n_users, n_movies), dtype=np.float32)
            mask = np.zeros((n_users, n_movies), dtype=np.float32)
            
            train_matrix[train_data['users'], train_data['movies']] = train_data['ratings']
            mask[train_data['users'], train_data['movies']] = 1.0
            
            try:
                # 凸方法1: SVT
                print("  训练凸方法 (SVT)...")
                svt_pred = nuclear_norm_svt_corrected(
                    train_matrix, mask, lambda_reg=lambda_reg, max_iter=30
                )
                svt_predictions = svt_pred[test_data['users'], test_data['movies']]
                svt_rmse = calculate_rmse(svt_predictions, test_data['ratings'])
                print(f"    SVT RMSE: {svt_rmse:.4f}")
                
                if svt_rmse < best_results['convex_svt']['rmse']:
                    best_results['convex_svt']['lambda'] = lambda_reg
                    best_results['convex_svt']['rmse'] = svt_rmse
                
            except Exception as e:
                print(f"    SVT失败: {e}")
            
            try:
                # 凸方法2: SoftImpute
                print("  训练凸方法 (SoftImpute)...")
                soft_pred = soft_impute_corrected(
                    train_matrix, mask, lambda_reg=lambda_reg, max_iter=30
                )
                soft_predictions = soft_pred[test_data['users'], test_data['movies']]
                soft_rmse = calculate_rmse(soft_predictions, test_data['ratings'])
                print(f"    SoftImpute RMSE: {soft_rmse:.4f}")
                
                if soft_rmse < best_results['convex_softimpute']['rmse']:
                    best_results['convex_softimpute']['lambda'] = lambda_reg
                    best_results['convex_softimpute']['rmse'] = soft_rmse
                
            except Exception as e:
                print(f"    SoftImpute失败: {e}")
            
            try:
                # 非凸方法: ALS
                print("  训练非凸方法 (ALS)...")
                als_model = CorrectedALSMatrixCompletion(
                    n_factors=10, reg=lambda_reg, max_iter=10, global_mean=global_mean
                )
                als_model.fit(train_data, n_users, n_movies)
                als_predictions = als_model.predict(test_data['users'], test_data['movies'])
                als_rmse = calculate_rmse(als_predictions, test_data['ratings'])
                print(f"    ALS RMSE: {als_rmse:.4f}")
                
                if als_rmse < best_results['als']['rmse']:
                    best_results['als']['lambda'] = lambda_reg
                    best_results['als']['rmse'] = als_rmse
                
            except Exception as e:
                print(f"    ALS失败: {e}")
            
            # 清理内存
            gc.collect()
        
        fold_idx += 1
    
    # 结果汇总
    print("\n" + "="*60)
    print("参数调优结果")
    print("="*60)
    print(f"基线RMSE: {baseline_rmse:.4f}")
    
    for method, results in best_results.items():
        if results['lambda'] is not None:
            print(f"\n{method} 最佳参数:")
            print(f"  最佳 λ: {results['lambda']}")
            print(f"  最佳RMSE: {results['rmse']:.4f}")
            improvement = ((baseline_rmse - results['rmse']) / baseline_rmse) * 100
            print(f"  相对于基线的改进: {improvement:.1f}%")
    
    return best_results, baseline_rmse

def run_final_experiment(mode='small'):
    """运行最终实验（使用最佳参数）"""
    if mode == 'small':
        sample_size = 50000
        n_folds = 5
        print("=" * 60)
        print("MovieLens 20M 最终实验（小规模）")
    else:
        sample_size = 100000
        n_folds = 3
        print("=" * 60)
        print("MovieLens 20M 最终实验（中等规模）")
    
    print("=" * 60)
    
    print("\n1. 加载数据...")
    ratings_matrix, ratings_df, n_users, n_movies = load_ratings_only(
        sample_size=sample_size
    )
    
    print(f"\n2. 数据统计:")
    print(f"   用户数: {n_users}")
    print(f"   电影数: {n_movies}")
    print(f"   评分数: {len(ratings_df)}")
    
    print(f"\n3. 准备{n_folds}折交叉验证...")
    cv_splits = create_cv_split(ratings_df, n_folds=n_folds)
    
    results = {
        'baseline': [],
        'convex_svt': [],
        'convex_softimpute': [],
        'als': []
    }
    
    # 基于之前实验的最佳参数
    best_params = {
        'convex_svt': 0.5,
        'convex_softimpute': 0.5,
        'als': 0.5
    }
    
    fold_idx = 1
    for train_idx, test_idx in cv_splits:
        print(f"\n{'='*40}")
        print(f"折 {fold_idx}/{n_folds}")
        print(f"{'='*40}")
        
        train_data, test_data = create_train_test_indices(ratings_df, train_idx, test_idx)
        
        print(f"  训练集大小: {len(train_data['users']):,}")
        print(f"  测试集大小: {len(test_data['users']):,}")
        
        # 基线方法
        global_mean = np.mean(train_data['ratings'])
        baseline_predictions = np.full(len(test_data['ratings']), global_mean)
        baseline_rmse = calculate_rmse(baseline_predictions, test_data['ratings'])
        results['baseline'].append(baseline_rmse)
        print(f"\n  基线RMSE: {baseline_rmse:.4f}")
        
        # 创建训练矩阵
        train_matrix = np.zeros((n_users, n_movies), dtype=np.float32)
        mask = np.zeros((n_users, n_movies), dtype=np.float32)
        
        train_matrix[train_data['users'], train_data['movies']] = train_data['ratings']
        mask[train_data['users'], train_data['movies']] = 1.0
        
        # 凸方法1: SVT
        try:
            print("\n  训练凸方法 (SVT)...")
            svt_pred = nuclear_norm_svt_corrected(
                train_matrix, mask, 
                lambda_reg=best_params['convex_svt'],
                max_iter=50 if mode == 'small' else 30
            )
            svt_predictions = svt_pred[test_data['users'], test_data['movies']]
            svt_rmse = calculate_rmse(svt_predictions, test_data['ratings'])
            results['convex_svt'].append(svt_rmse)
            print(f"    SVT RMSE: {svt_rmse:.4f}")
            print(f"    相对于基线的改进: {((baseline_rmse - svt_rmse) / baseline_rmse * 100):.1f}%")
        except Exception as e:
            print(f"    SVT失败: {e}")
            results['convex_svt'].append(float('nan'))
        
        # 凸方法2: SoftImpute
        try:
            print("\n  训练凸方法 (SoftImpute)...")
            soft_pred = soft_impute_corrected(
                train_matrix, mask,
                lambda_reg=best_params['convex_softimpute'],
                max_iter=50 if mode == 'small' else 30
            )
            soft_predictions = soft_pred[test_data['users'], test_data['movies']]
            soft_rmse = calculate_rmse(soft_predictions, test_data['ratings'])
            results['convex_softimpute'].append(soft_rmse)
            print(f"    SoftImpute RMSE: {soft_rmse:.4f}")
            print(f"    相对于基线的改进: {((baseline_rmse - soft_rmse) / baseline_rmse * 100):.1f}%")
        except Exception as e:
            print(f"    SoftImpute失败: {e}")
            results['convex_softimpute'].append(float('nan'))
        
        # 非凸方法: ALS
        try:
            print("\n  训练非凸方法 (ALS)...")
            als_model = CorrectedALSMatrixCompletion(
                n_factors=10, 
                reg=best_params['als'], 
                max_iter=10, 
                global_mean=global_mean
            )
            als_model.fit(train_data, n_users, n_movies)
            als_predictions = als_model.predict(test_data['users'], test_data['movies'])
            als_rmse = calculate_rmse(als_predictions, test_data['ratings'])
            results['als'].append(als_rmse)
            print(f"    ALS RMSE: {als_rmse:.4f}")
            print(f"    相对于基线的改进: {((baseline_rmse - als_rmse) / baseline_rmse * 100):.1f}%")
        except Exception as e:
            print(f"    ALS失败: {e}")
            results['als'].append(float('nan'))
        
        # 清理内存
        del train_matrix, mask
        gc.collect()
        
        fold_idx += 1
    
    # 结果汇总
    print("\n" + "="*60)
    print("最终实验结果汇总")
    print("="*60)
    
    # 过滤NaN值并计算统计量
    for method in results:
        valid_scores = [s for s in results[method] if not np.isnan(s)]
        if valid_scores:
            mean_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
            
            print(f"\n{method}:")
            print(f"  平均RMSE: {mean_score:.4f}")
            print(f"  标准差: {std_score:.4f}")
            print(f"  各折RMSE: {[f'{x:.4f}' for x in valid_scores]}")
            
            if method != 'baseline' and results['baseline']:
                baseline_mean = np.mean([s for s in results['baseline'] if not np.isnan(s)])
                improvement = ((baseline_mean - mean_score) / baseline_mean) * 100
                print(f"  相对于基线的平均改进: {improvement:.1f}%")
    
    return results

# ==================== 6. 主程序入口 ====================
if __name__ == "__main__":
    print("选择实验模式:")
    print("1. 参数调优实验（小规模，寻找最佳参数）")
    print("2. 最终实验（小规模，使用最佳参数）")
    print("3. 最终实验（中等规模，使用最佳参数）")
    
    choice = input("请输入选择 (1, 2, 或3): ").strip()
    
    try:
        if choice == "1":
            best_results, baseline_rmse = run_parameter_tuning_experiment()
            
            # 保存参数调优结果
            import os
            import datetime
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"parameter_tuning_results_{timestamp}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("MovieLens 20M 参数调优结果\n")
                f.write(f"实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n")
                f.write(f"基线RMSE: {baseline_rmse:.4f}\n\n")
                
                for method, results in best_results.items():
                    if results['lambda'] is not None:
                        f.write(f"{method}:\n")
                        f.write(f"  最佳 λ: {results['lambda']}\n")
                        f.write(f"  最佳RMSE: {results['rmse']:.4f}\n")
                        improvement = ((baseline_rmse - results['rmse']) / baseline_rmse) * 100
                        f.write(f"  相对于基线的改进: {improvement:.1f}%\n\n")
            
            print(f"\n参数调优结果已保存到 {output_file}")
            
        elif choice == "2":
            results = run_final_experiment(mode='small')
            
            # 保存最终实验结果
            import os
            import datetime
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"final_results_small_{timestamp}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("MovieLens 20M 最终实验结果（小规模）\n")
                f.write(f"实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n")
                
                for method, scores in results.items():
                    valid_scores = [s for s in scores if not np.isnan(s)]
                    if valid_scores:
                        f.write(f"{method}:\n")
                        f.write(f"  平均RMSE: {np.mean(valid_scores):.4f}\n")
                        f.write(f"  各折RMSE: {valid_scores}\n\n")
            
            print(f"\n最终实验结果已保存到 {output_file}")
            
        elif choice == "3":
            results = run_final_experiment(mode='medium')
            
            # 保存最终实验结果
            import os
            import datetime
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"final_results_medium_{timestamp}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("MovieLens 20M 最终实验结果（中等规模）\n")
                f.write(f"实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n")
                
                for method, scores in results.items():
                    valid_scores = [s for s in scores if not np.isnan(s)]
                    if valid_scores:
                        f.write(f"{method}:\n")
                        f.write(f"  平均RMSE: {np.mean(valid_scores):.4f}\n")
                        f.write(f"  各折RMSE: {valid_scores}\n\n")
            
            print(f"\n最终实验结果已保存到 {output_file}")
            
        else:
            print("无效选择，默认运行参数调优实验")
            best_results, baseline_rmse = run_parameter_tuning_experiment()
        
        print("\n实验完成！")
        
    except Exception as e:
        print(f"运行实验时出错: {e}")
        import traceback
        traceback.print_exc()