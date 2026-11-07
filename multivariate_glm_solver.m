% ==============================================================================
% Ads Marketing Lifecycle Analytics: IRLS Generalized Linear Model (GLM),
% SVD Multivariate Analysis & Neyman Stratified Allocation Solver in MATLAB
% ==============================================================================
function results = multivariate_glm_solver(X, y_conv, y_spend, strata_id, n_sample)
    % 1. Binomial Logit GLM via Iteratively Reweighted Least Squares (IRLS)
    [n, p] = size(X);
    beta_logit = zeros(p, 1);
    reg = 1e-2 * eye(p); reg(1,1) = 0;
    for iter = 1:25
        eta = max(min(X * beta_logit, 18), -18);
        mu = 1 ./ (1 + exp(-eta));
        w = max(mu .* (1 - mu), 1e-6);
        grad = X' * (y_conv - mu) - reg * beta_logit;
        H = X' * bsxfun(@times, w, X) + reg;
        step = H \ grad;
        beta_logit = beta_logit + step;
        if max(abs(step)) < 1e-6
            break;
        end
    end

    % 2. Multivariate SVD / Principal Component Analysis
    Z = zscore(X(:, 2:end));
    [~, S, V] = svd(Z, 'econ');
    eigvals = diag(S).^2 / (n - 1);
    explained_ratio = cumsum(eigvals) / sum(eigvals);

    % 3. Neyman Optimal Stratified Allocation
    unique_strata = unique(strata_id);
    H_count = numel(unique_strata);
    W_h = zeros(H_count, 1);
    S_h = zeros(H_count, 1);
    for h = 1:H_count
        mask = (strata_id == unique_strata(h));
        W_h(h) = sum(mask) / n;
        S_h(h) = std(y_spend(mask));
    end
    n_h = round(n_sample * (W_h .* S_h) / sum(W_h .* S_h));

    results = struct( ...
        'beta_logit', beta_logit, ...
        'pca_loadings', V, ...
        'pca_cumulative_explained', explained_ratio, ...
        'neyman_strata_allocation', n_h ...
    );
end
