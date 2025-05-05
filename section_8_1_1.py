# 8.1.1 Select relevant features for validation data and add constant
print("\n8.1.1 Adding constant to validation data")

# Select the relevant features for validation data
X_val_model = X_val_dummies[col]
print(f"Selected {X_val_model.shape[1]} features for validation data")
print(f"First few selected features: {col[:5]}")

# Add constant to X_validation
X_val_sm = sm.add_constant(X_val_model)
print(f"Validation data shape after adding constant: {X_val_sm.shape}")
