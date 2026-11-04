# DRATS estimator: one or two binary treatment decisions
# Based on the DRATS estimating equations in SimulationsObjective3_TwoDecisionStage.R
# Continuous outcomes; complete data; binary treatments 0/1.

# Fit a Super Learner and predict on a validation set.
dr_fit_predict = function(train, valid, y, predictors, family, library) {
  fit = SuperLearner::SuperLearner(
    Y = train[[y]], X = train[, predictors, drop = FALSE],
    family = if (family == "binomial") binomial() else gaussian(),
    SL.library = library, method = "method.NNLS", verbose = FALSE)
  as.numeric(predict(fit, newdata = valid[, predictors, drop = FALSE])$pred)
}

# Cross-fitted conditional expectation with optional treatment interventions.
dr_crossfit = function(data, response, predictors, fold_id,
                        family = "gaussian", library = c("SL.mean", "SL.glm"),
                        interventions = list()) {
  out = vector("list", length(interventions) + 1L)
  names(out) = c("observed", names(interventions))
  for (nm in names(out)) out[[nm]] = rep(NA_real_, nrow(data))
  for (v in sort(unique(fold_id))) {
    tr = data[fold_id != v, , drop = FALSE]
    va = data[fold_id == v, , drop = FALSE]
    fit = SuperLearner::SuperLearner(
      Y = tr[[response]], X = tr[, predictors, drop = FALSE],
      family = if (family == "binomial") binomial() else gaussian(),
      SL.library = library, method = "method.NNLS", verbose = FALSE)
    idx = which(fold_id == v)
    out$observed[idx] = as.numeric(predict(fit, newdata = va[, predictors, drop = FALSE])$pred)
    for (nm in names(interventions)) {
      new = va
      for (variable in names(interventions[[nm]])) {
        values = interventions[[nm]][[variable]]
        new[[variable]] = if (length(values) == 1L) values else values[idx]
      }
      out[[nm]][idx] = as.numeric(predict(fit, newdata = new[, predictors, drop = FALSE])$pred)
    }
  }
  out
}

# EIF-based linear projection, with covariance computed from centered scores.
dr_project = function(L, p, score, alpha = 0.05) {
  w = p * (1 - p)
  M = crossprod(L, L * w) / nrow(L)
  psi = as.vector(solve(M, colMeans(L * (w * score))))
  # Influence-function contribution of the estimating equation (conditional on nuisances).
  influence = (L * (w * score) - (L %*% psi) * L * w) %*% solve(M)
  variance = stats::cov(influence) / nrow(L)
  se = sqrt(diag(variance))
  z = qnorm(1 - alpha / 2)
  names(psi) = colnames(L)
  names(se) = colnames(L)
  list(psi = psi, se = se, variance = variance,
       ci = cbind(lower = psi - z * se, upper = psi + z * se),
       influence = influence)
}

# Main public function.
drats = function(data, outcome, treat.names, H.vars, blip.formulas,
                  folds = 2, SL.library = c("SL.glm", "SL.randomForest", "SL.nnet", "SL.glm.interaction"),
                  alpha = 0.05, ps.truncation = "none", seed = 123, verbose = TRUE) {
  if (!requireNamespace("SuperLearner", quietly = TRUE)) stop("Install 'SuperLearner'.")
  K = length(treat.names)
  if (!K %in% 1:2 || length(H.vars) != K || length(blip.formulas) != K)
    stop("Provide matching arguments for one or two decision stages.")
  if (!is.data.frame(data) || !is.numeric(data[[outcome]])) stop("Continuous numeric outcome required.")
  if (folds < 2 || folds > nrow(data) || alpha <= 0 || alpha >= 1 ||
      !ps.truncation %in% c("none", "gruber")) stop("Invalid folds, alpha or ps.truncation (none/gruber).")
  variables = unique(c(outcome, treat.names, unlist(H.vars),
                        unlist(lapply(blip.formulas, all.vars))))
  if (!all(variables %in% names(data))) stop("Some requested variables are missing.")
  if (anyNA(data[, variables, drop = FALSE])) stop("Missing values are not supported.")
  for (a in treat.names) if (!all(sort(unique(data[[a]])) == c(0, 1)))
    stop("Each treatment must contain both 0 and 1.")
  if (K == 2 && !treat.names[1] %in% H.vars[[2]])
    stop("Stage-2 history must include stage-1 treatment.")
  set.seed(seed)
  n = nrow(data)
  fold_id = sample(rep(seq_len(folds), length.out = n))
  Y = data[[outcome]]
  L = lapply(blip.formulas, model.matrix, data = data)
  if (any(vapply(L, function(x) qr(x)$rank < ncol(x), logical(1))))
    stop("Blip design is rank deficient.")
  # "none": raw propensity predictions; "gruber": Gruber bound.
  # The sample size n is the full analysis sample, as in the requested rule.
  gruber_bound = 5 / (sqrt(n) * log(n))
  if (ps.truncation == "gruber" && gruber_bound >= 0.5)
    stop("Gruber bound is >= 0.5; increase sample size.")
  bound = function(p) {
    if (any(!is.finite(p))) stop("Non-finite propensity predictions.")
    if (ps.truncation == "gruber") p = pmin(1 - gruber_bound, pmax(gruber_bound, p))
    if (any(p <= 0 | p >= 1))
      stop("Propensities equal to 0 or 1. Use ps.truncation = 'gruber'.")
    p
  }
  A = lapply(treat.names, function(x) data[[x]])
  stage = vector("list", K)
  p = vector("list", K)
  for (j in seq_len(K)) {
    p[[j]] = bound(dr_crossfit(data, treat.names[j], H.vars[[j]], fold_id,
                              "binomial", SL.library)$observed)
  }
  # Last stage: DRATS augmented conditional mean contrast.
  j = K
  a = treat.names[j]
  nuisance2 = dr_crossfit(data, outcome, c(a, H.vars[[j]]), fold_id,
                           library = SL.library,
                           interventions = list(one = setNames(list(1), a),
                                                zero = setNames(list(0), a)))
  m1 = nuisance2$one; m0 = nuisance2$zero
  contrast = A[[j]] / p[[j]] * (Y - m1) + m1 -
              (1 - A[[j]]) / (1 - p[[j]]) * (Y - m0) - m0
  stage[[j]] = dr_project(L[[j]], p[[j]], contrast, alpha)
  stage[[j]]$optimal_treatment = as.integer(L[[j]] %*% stage[[j]]$psi > 0)
  stage[[j]]$propensity = p[[j]]
  if (verbose) message("DRATS: completed stage ", j)
  if (K == 2) {
    # First-stage equation follows the six EIF components of Objective 3.
    a1 = treat.names[1]; a2 = treat.names[2]
    opt2 = stage[[2]]$optimal_treatment
    base2 = H.vars[[2]]
    ps2_cf = dr_crossfit(data, a2, base2, fold_id, "binomial", SL.library,
      interventions = list(a1one = setNames(list(1), a1),
                           a1zero = setNames(list(0), a1)))
    p21 = bound(ps2_cf$a1one); p20 = bound(ps2_cf$a1zero)
    # Outcome regression under factual A2 and estimated optimal A2,
    # with A1 counterfactually set to 0 or 1 as in the simulation.
    y2 = dr_crossfit(data, outcome, c(a2, base2), fold_id,
      library = SL.library,
      interventions = list(opt = setNames(list(opt2), a2),
        opt1 = setNames(list(opt2, 1), c(a2, a1)),
        opt0 = setNames(list(opt2, 0), c(a2, a1)),
        fact1 = setNames(list(A[[2]], 1), c(a2, a1)),
        fact0 = setNames(list(A[[2]], 0), c(a2, a1))))
    # Stage-1 regression of the observed outcome, under A1 interventions.
    y1 = dr_crossfit(data, outcome, c(a1, H.vars[[1]]), fold_id,
      library = SL.library,
      interventions = list(one = setNames(list(1), a1),
                           zero = setNames(list(0), a1)))
    # Second layer of regressions: regress out-of-fold stage-2 predictions
    # on A1 and baseline history, as in the simulation program.
    nested = data
    nested$.opt = y2$opt
    nested$.fact = y2$observed
    Eopt = dr_crossfit(nested, ".opt", c(a1, H.vars[[1]]), fold_id,
      library = SL.library,
      interventions = list(one = setNames(list(1), a1),
                           zero = setNames(list(0), a1)))
    Efact = dr_crossfit(nested, ".fact", c(a1, H.vars[[1]]), fold_id,
      library = SL.library,
      interventions = list(one = setNames(list(1), a1),
                           zero = setNames(list(0), a1)))
    a1obs = A[[1]]; a2obs = A[[2]]; p1 = p[[1]]
    match_opt = as.numeric(a2obs == opt2)
    fopt1 = ifelse(opt2 == 1, p21, 1 - p21)
    fopt0 = ifelse(opt2 == 1, p20, 1 - p20)
    t0 = a1obs / p1 * (Y - y1$one) + y1$one
    t1 = match_opt * a1obs / (fopt1 * p1) * (Y - y2$opt1) +
          a1obs / p1 * (y2$opt1 - Eopt$one) + Eopt$one
    t2 = a1obs / p1 * (Y - Efact$one) + Efact$one
    t3 = (1 - a1obs) / (1 - p1) * (Y - y1$zero) + y1$zero
    t4 = match_opt * (1 - a1obs) / (fopt0 * (1 - p1)) * (Y - y2$opt0) +
          (1 - a1obs) / (1 - p1) * (y2$opt0 - Eopt$zero) + Eopt$zero
    t5 = (1 - a1obs) / (1 - p1) * (Y - Efact$zero) + Efact$zero
    score1 = t0 + t1 - t2 - t3 - t4 + t5
    stage[[1]] = dr_project(L[[1]], p1, score1, alpha)
    stage[[1]]$optimal_treatment = as.integer(L[[1]] %*% stage[[1]]$psi > 0)
    stage[[1]]$propensity = p1
    if (verbose) message("DRATS: completed stage 1")
  }
  names(stage) = paste0("stage", seq_len(K))
  list(call = match.call(), K = K, psi = lapply(stage, `[[`, "psi"),
       se = lapply(stage, `[[`, "se"), ci = lapply(stage, `[[`, "ci"),
       variance = lapply(stage, `[[`, "variance"),
       opt.treat = lapply(stage, `[[`, "optimal_treatment"),
       propensity = lapply(stage, `[[`, "propensity"),
       stages = stage, fold_id = fold_id,
       settings = list(ps.truncation = ps.truncation, gruber_bound = if (ps.truncation == "gruber") gruber_bound else NULL,
                       SL.library = SL.library, folds = folds))
}

