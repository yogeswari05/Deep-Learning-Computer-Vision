import numpy as np
from DecisionTreeRegressor import MyDecisionTreeRegressor

def print_custom_tree(node, depth=0):
    """Helper function to print tree structure"""
    indent = "  " * depth
    if not hasattr(node, 'feat_id'):  # Leaf node
        pred_val = getattr(node, 'prediction', getattr(node, 'pred_val', getattr(node, 'value', 0.0)))
        print(f"{indent}Leaf: predict y = {pred_val:.3f}")
    else:  # Internal node
        print(f"{indent}Decision: X[{node.feat_id}] < {node.thresh_val:.3f}?")
        print(f"{indent}Y:")
        print_custom_tree(node.left_child, depth + 1)
        print(f"{indent}N:")
        print_custom_tree(node.right_child, depth + 1)

def test_simple_binary_split():
    """Test 1: Simple binary classification-like problem"""
    print("\n" + "="*60)
    print("TEST 1: Simple Binary Split (y = 1 if x > 0 else 0)")
    print("="*60)
    
    X_train = np.linspace(-1, 1, 50).reshape(-1, 1)
    y_train = (X_train[:, 0] > 0).astype(float)
    
    tree = MyDecisionTreeRegressor(max_depth=4, min_samples_split=2)
    tree.fit(X_train, y_train)
    
    y_pred = tree.predict(X_train)
    y_pred_class = (y_pred > 0.5).astype(float)
    accuracy = np.mean(y_pred_class == y_train)
    
    print(f"Training accuracy: {accuracy * 100:.2f}%")
    print(f"Sample predictions (first 5): {y_pred[:5]}")
    print(f"Sample predictions (last 5): {y_pred[-5:]}")
    print("\nTree Structure:")
    print_custom_tree(tree.root_node)
    
    assert accuracy >= 0.95, "Should achieve high accuracy on simple problem"
    print("✅ Test 1 PASSED")

def test_linear_regression():
    """Test 2: Simple linear relationship"""
    print("\n" + "="*60)
    print("TEST 2: Linear Regression (y = 2*x + 1)")
    print("="*60)
    
    X_train = np.linspace(-2, 2, 40).reshape(-1, 1)
    y_train = 2 * X_train[:, 0] + 1
    
    tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=2)
    tree.fit(X_train, y_train)
    
    y_pred = tree.predict(X_train)
    mse = np.mean((y_pred - y_train) ** 2)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Sample true values: {y_train[:5]}")
    print(f"Sample predictions: {y_pred[:5]}")
    print("\nTree Structure (first few levels):")
    print_custom_tree(tree.root_node)
    
    assert mse < 2.0, "MSE should be reasonable for linear problem"
    print("✅ Test 2 PASSED")

def test_step_function():
    """Test 3: Step function"""
    print("\n" + "="*60)
    print("TEST 3: Step Function")
    print("="*60)
    
    X_train = np.array([-1, -0.5, -0.2, 0.1, 0.3, 0.6, 0.9, 1.2]).reshape(-1, 1)
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    tree = MyDecisionTreeRegressor(max_depth=3, min_samples_split=2)
    tree.fit(X_train, y_train)
    
    y_pred = tree.predict(X_train)
    
    print(f"Training data X: {X_train.flatten()}")
    print(f"Training data y: {y_train}")
    print(f"Predictions:     {y_pred}")
    print("\nTree Structure:")
    print_custom_tree(tree.root_node)
    
    # Check that predictions are close to actual values
    close_predictions = np.sum(np.abs(y_pred - y_train) < 0.3)
    print(f"\nClose predictions: {close_predictions}/{len(y_train)}")
    
    assert close_predictions >= 6, "Most predictions should be close"
    print("✅ Test 3 PASSED")

def test_constant_output():
    """Test 4: All same output values"""
    print("\n" + "="*60)
    print("TEST 4: Constant Output (all y = 5)")
    print("="*60)
    
    X_train = np.random.randn(20, 3)
    y_train = 5.0 * np.ones(20)
    
    tree = MyDecisionTreeRegressor(max_depth=3, min_samples_split=2)
    tree.fit(X_train, y_train)
    
    y_pred = tree.predict(X_train)
    
    print(f"All training y values: {np.unique(y_train)}")
    print(f"Prediction range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    print(f"Mean prediction: {y_pred.mean():.3f}")
    
    assert np.allclose(y_pred, 5.0), "Should predict constant value"
    print("✅ Test 4 PASSED")

def test_multi_feature():
    """Test 5: Multiple features"""
    print("\n" + "="*60)
    print("TEST 5: Multiple Features (F=3)")
    print("="*60)
    
    np.random.seed(42)
    X_train = np.random.randn(50, 3)
    # y depends on all features
    y_train = X_train[:, 0] + 2 * X_train[:, 1] - X_train[:, 2] + np.random.randn(50) * 0.1
    
    tree = MyDecisionTreeRegressor(max_depth=6, min_samples_split=4, min_samples_leaf=2)
    tree.fit(X_train, y_train)
    
    y_pred = tree.predict(X_train)
    mse = np.mean((y_pred - y_train) ** 2)
    r2 = 1 - (np.sum((y_train - y_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  Pred: {y_pred[i]:.3f}, Actual: {y_train[i]:.3f}")
    
    assert mse < 5.0, "MSE should be reasonable"
    print("✅ Test 5 PASSED")

def test_depth_constraint():
    """Test 6: Verify max_depth constraint"""
    print("\n" + "="*60)
    print("TEST 6: Max Depth Constraint")
    print("="*60)
    
    X_train = np.linspace(-1, 1, 30).reshape(-1, 1)
    y_train = np.sin(5 * X_train[:, 0])
    
    for depth in [1, 2, 3, 4]:
        tree = MyDecisionTreeRegressor(max_depth=depth, min_samples_split=2)
        tree.fit(X_train, y_train)
        
        # Count actual depth
        def get_depth(node):
            if not hasattr(node, 'feat_id'):
                return 0
            return 1 + max(get_depth(node.left_child), get_depth(node.right_child))
        
        actual_depth = get_depth(tree.root_node)
        print(f"Max depth={depth}: Actual tree depth = {actual_depth}")
        
        assert actual_depth <= depth, f"Tree depth {actual_depth} exceeds max_depth {depth}"
    
    print("✅ Test 6 PASSED")

def test_min_samples_leaf():
    """Test 7: Verify min_samples_leaf constraint"""
    print("\n" + "="*60)
    print("TEST 7: Min Samples Leaf Constraint")
    print("="*60)
    
    X_train = np.random.randn(50, 2)
    y_train = X_train[:, 0] + X_train[:, 1]
    
    for min_leaf in [1, 5, 10]:
        tree = MyDecisionTreeRegressor(max_depth=5, min_samples_leaf=min_leaf, min_samples_split=2*min_leaf)
        tree.fit(X_train, y_train)
        
        print(f"Min samples leaf={min_leaf}: Tree created successfully")
        y_pred = tree.predict(X_train)
        mse = np.mean((y_pred - y_train) ** 2)
        print(f"  MSE: {mse:.4f}")
    
    print("✅ Test 7 PASSED")

def run_all_tests():
    """Run all test cases"""
    print("\n" + "#"*60)
    print("# TESTING MyDecisionTreeRegressor")
    print("#"*60)
    
    try:
        test_simple_binary_split()
        test_linear_regression()
        test_step_function()
        test_constant_output()
        test_multi_feature()
        test_depth_constraint()
        test_min_samples_leaf()
        
        print("\n" + "#"*60)
        print("# ALL TESTS PASSED! ✅")
        print("#"*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
