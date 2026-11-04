
# ======================================================================
# dwols_sl
# Dynamic Weighted Ordinary Least Squares using SuperLearner
# with Gruber truncation and m-out-of-n bootstrap confidence intervals
#
# Author: Kossi Clément Trenou et al.
# Version: 2026
# ======================================================================


# Arguments:
# ----------
# data :
#   A data frame containing the outcome, treatment variables,
#   history variables, and all covariates used in the
#   treatment-free and blip models.
#
# outcome :
#   A character string giving the name of the outcome variable.
#   The outcome must be numeric.
#
# treat.names :
#   A character vector containing the names of the treatment
#   variables, in chronological order.
#   - For one decision point: c("A1")
#   - For two decision points: c("A1", "A2")
#   Treatments must be binary and coded as 0/1.
#
# H.vars :
#   A list of character vectors.
#   Each element contains the names of the history variables
#   used to estimate the propensity score at a given stage.
#   Examples:
#   - One stage: list(c("X11","X12","X13"))
#   - Two stages: list(c("X11","X12"), c("A1","X11","X12","X21","X22"))
#
# tf.formulas :
#   A list of formula objects defining the treatment-free model
#   at each stage.
#   Only the right-hand side should be supplied.
#   Examples:
#   - list(~ X11 + X12 + X13)
#   - list(~ X11 + X12, ~ X21 + X22 + A1 + X11 + X12)
#
# blip.formulas :
#   A list of formula objects defining the blip model at each stage.
#   Only the right-hand side should be supplied.
#   Include 1 if an intercept in the blip model is desired.
#   Examples:
#   - list(~ 1 + X11 + X12 + X13)
#   - list(~ 1 + X11 + X12, ~ 1 + X21 + X22 + X23)
#
# weight :
#   A character string specifying the type of weights used in dWOLS.
#   - "default" : overlap weights, i.e. |A - p_hat| by kossi clément trenou & al.
#   Default is "default".
#
# use_cross_fitting :
#   Logical. If TRUE, cross-fitting is used to estimate the
#   propensity scores. If FALSE, the propensity scores are
#   estimated on the full sample.
#   Default is TRUE.
#
# folds :
#   Integer giving the number of folds used for cross-fitting.
#   Must be >= 2 when use_cross_fitting = TRUE.
#   Default is 2.
#
# SL.library :
#   A character vector specifying the SuperLearner library
#   used for propensity score estimation.
#   Example:
#   c("SL.glm", "SL.randomForest", "SL.nnet", "SL.glm.interaction")
#
# SL.method :
#   A character string specifying the SuperLearner metalearner.
#   Example: "method.NNLS"
#
# seed :
#   Optional numeric value used for reproducibility.
#   If NULL, no seed is set.
#
# return_fits :
#   Logical. If TRUE, the fitted dWOLS models and the fitted
#   propensity score objects are returned in the output.
#   If FALSE, only the main estimation results are returned.
#   Default is TRUE.
#
# verbose :
#   Logical. If TRUE, progress messages are displayed during
#   estimation. If FALSE, the function runs silently.
#   Default is TRUE.
#
# Value:
# ------
# A list containing:
#
# call :
#   The matched function call.
#
# K :
#   The number of decision stages.
#
# outcome :
#   The name of the outcome variable.
#
# treat.names :
#   The names of the treatment variables.
#
# psi :
#   A list containing the estimated blip parameters at each stage.
#
# psi.vector :
#   A single vector containing all estimated blip parameters
#   concatenated across stages.
#
# opt.treat :
#   A list containing the estimated optimal treatment decision
#   for each individual at each stage.
#
# propensity :
#   A list containing the estimated propensity scores at each stage.
#
# weights :
#   A list containing the dWOLS weights at each stage.
#
# mu :
#   A list containing the estimated stage-specific regret contributions.
#
# regret :
#   A vector containing the estimated total regret for each individual.
#
# opt.Y :
#   A vector containing the estimated optimal outcome for each individual.
#
# blip.design :
#   A list containing the design matrices associated with the
#   blip models at each stage.
#
# stage.formulas :
#   A list containing the weighted regression formulas used
#   internally at each stage.
#
# settings :
#   A list containing the main estimation settings used in the call.
#
# models :
#   Returned only if return_fits = TRUE.
#   A list of fitted weighted least squares models at each stage.
#
# ps_fits :
#   Returned only if return_fits = TRUE.
#   A list containing the fitted SuperLearner objects used for
#   propensity score estimation.
#
# Details:
# --------
# This function implements dynamic weighted ordinary least squares
# (dWOLS) using SuperLearner for propensity score estimation only.
# It is designed for one-stage or two-stage treatment regimes.
#
# At each stage:
#   1. The propensity score is estimated using SuperLearner.
#   2. The dWOLS weights are computed.
#   3. A weighted least squares model is fitted to estimate the blip.
#   4. The optimal treatment rule is derived from the estimated blip.
#
# The function works backward from the last stage to the first stage.
#
# Important notes:
# ----------------
# - Treatments must be binary and coded 0/1.
# - Missing values are not allowed.
# - The outcome must be numeric.
# - This implementation currently supports only 1 or 2 stages.
# - SuperLearner is used only for propensity score estimation.
# - The final blip estimation step is performed using weighted
#   linear regression.


dwols_sl = function(
    data,                              # data frame contenant toutes les variables
    outcome,                           # nom de la variable de résultat Y
    treat.names,                       # noms des traitements : c("A1") ou c("A1", "A2")
    H.vars,                            # liste des historiques H_j pour chaque temps j
    tf.formulas,                       # liste des formules treatment-free m_j(H_j)
    blip.formulas,                     # liste des formules blip L_j(H_j)^T psi_j
    weight = "default",                # poids dWOLS : uniquement "default" = |A - p_hat|
    use_cross_fitting = TRUE,          # TRUE = cross-fitting pour la propension
    folds = 2,                         # nombre de folds
    SL.library = c(
      "SL.glm",
      "SL.randomForest",
      "SL.nnet",
      "SL.glm.interaction"
    ),                                 # bibliothèque SuperLearner
    SL.method = "method.NNLS",         # méta-learner SuperLearner
    seed = NULL,                       # graine pour reproductibilité
    return_fits = TRUE,                # TRUE = retourner les objets ajustés
    verbose = TRUE,                    # TRUE = afficher les messages

    # Options pour les intervalles de confiance
    bootstrap = FALSE,                 # TRUE = calculer les IC par m-out-of-n bootstrap
    B = 5,                           # nombre de réplications bootstrap
    q = 0.95,                          # facteur de décroissance des tailles m_j
    alpha = 0.05,                      # niveau alpha pour IC à 1-alpha
    bootstrap_seed = NULL              # graine spécifique au bootstrap
) {

  # --------------------------------------------------------------------
  # 0. Vérification du package requis
  # --------------------------------------------------------------------
  # SuperLearner est utilisé pour estimer les propensions :
  #   p_j(H_j) = P(A_j = 1 | H_j)
  # --------------------------------------------------------------------

  if (!requireNamespace("SuperLearner", quietly = TRUE)) {
    stop("Le package 'SuperLearner' doit être installé.", call. = FALSE)
  }


  # --------------------------------------------------------------------
  # 1. Fonctions utilitaires internes
  # --------------------------------------------------------------------

  # Fonction d'arrêt propre sans afficher l'appel complet.
  .stop = function(...) stop(..., call. = FALSE)

  # Fonction d'avertissement propre.
  .warn = function(...) warning(..., call. = FALSE)

  # Vérifie qu'un objet est une chaîne de caractères de longueur 1.
  .is_scalar_character = function(x) {
    is.character(x) && length(x) == 1L && !is.na(x)
  }

  # Vérifie qu'un objet est un booléen TRUE/FALSE valide.
  .is_flag = function(x) {
    is.logical(x) && length(x) == 1L && !is.na(x)
  }

  # Vérifie qu'un argument est une liste de formules.
  .check_formula_list = function(x, arg_name) {

    if (!is.list(x)) {
      .stop(arg_name, " doit être une liste de formules.")
    }

    if (length(x) == 0L) {
      .stop(arg_name, " ne peut pas être vide.")
    }

    ok = vapply(x, inherits, logical(1), what = "formula")

    if (!all(ok)) {
      .stop(arg_name, " doit contenir uniquement des objets formula.")
    }
  }


  # --------------------------------------------------------------------
  # 2. Troncature de Gruber
  # --------------------------------------------------------------------
  # La troncature empêche les propensions estimées d'être trop proches
  # de 0 ou de 1.
  #
  # Borne de Gruber :
  #   b_n = 5 / (sqrt(n) log(n))
  #
  # Propension tronquée :
  #   p_trunc = min(max(p, b_n), 1 - b_n)
  # --------------------------------------------------------------------

  truncate_gruber = function(p, n) {

    # Borne de Gruber.
    b = 5 / (sqrt(n) * log(n))

    # Troncature.
    p = pmax(p, b)
    p = pmin(p, 1 - b)

    return(p)
  }


  # --------------------------------------------------------------------
  # 3. Estimation de la propension score par SuperLearner
  # --------------------------------------------------------------------

  .fit_ps = function(A, H) {

    # Cette fonction ajuste un SuperLearner pour estimer :
    #   P(A = 1 | H)
    #
    # Ici :
    #   Y      = A
    #   X      = H
    #   family = binomial(), car A est binaire.

    SuperLearner::SuperLearner(
      Y = A,
      X = as.data.frame(H),
      family = stats::binomial(),
      SL.library = SL.library,
      method = SL.method,
      verbose = FALSE
    )
  }


  # --------------------------------------------------------------------
  # 4. Prédiction des propensions avec troncature de Gruber
  # --------------------------------------------------------------------

  .predict_ps = function(fit, newdata) {

    # Prédiction de :
    #   p_hat = P_hat(A = 1 | H)

    p = as.numeric(
      predict(
        fit,
        newdata = as.data.frame(newdata)
      )$pred
    )

    # Application de la troncature de Gruber.
    p = truncate_gruber(
      p = p,
      n = nrow(newdata)
    )

    return(p)
  }


  # --------------------------------------------------------------------
  # 5. Cross-fitting pour la propension
  # --------------------------------------------------------------------
  # L'échantillon est séparé en folds.
  # Pour chaque fold, le SuperLearner est entraîné sur les autres folds,
  # puis prédit sur le fold laissé de côté.
  # --------------------------------------------------------------------

  .crossfit_ps = function(A, H, folds, seed_local = NULL) {

    H = as.data.frame(H)
    n = nrow(H)

    if (!is.null(seed_local)) {
      set.seed(seed_local)
    }

    # Attribution des observations aux folds.
    fold_id = sample(rep(seq_len(folds), length.out = n))

    # Vecteur de propensions cross-fittées.
    p_hat = rep(NA_real_, n)

    # Liste des modèles ajustés.
    fit_list = vector("list", folds)

    for (v in seq_len(folds)) {

      # Indices d'apprentissage.
      idx_train = which(fold_id != v)

      # Indices de validation.
      idx_valid = which(fold_id == v)

      # Ajustement sur l'échantillon d'apprentissage.
      fit_v = .fit_ps(
        A = A[idx_train],
        H = H[idx_train, , drop = FALSE]
      )

      fit_list[[v]] = fit_v

      # Prédiction sur l'échantillon de validation.
      p_hat[idx_valid] = .predict_ps(
        fit = fit_v,
        newdata = H[idx_valid, , drop = FALSE]
      )
    }

    list(
      p_hat = p_hat,
      fold_id = fold_id,
      fit = fit_list
    )
  }


  # --------------------------------------------------------------------
  # 6. Calcul des poids dWOLS
  # --------------------------------------------------------------------
  # Poids utilisé :
  #   w_j = |A_j - p_hat_j(H_j)|
  #
  # Il s'agit du poids dWOLS par défaut.
  # --------------------------------------------------------------------

  .compute_weights = function(A, p_hat) {

    if (anyNA(p_hat)) {
      .stop("p_hat contient des NA.")
    }

    if (any(p_hat <= 0 | p_hat >= 1)) {
      .stop("p_hat doit être strictement dans (0,1).")
    }

    # Poids overlap / dWOLS.
    w = abs(A - p_hat)

    return(w)
  }


  # --------------------------------------------------------------------
  # 7. Construction de la formule de régression dWOLS
  # --------------------------------------------------------------------
  # Le modèle ajusté au temps j est :
  #
  #   Y^(j) = m_j(H_j) + A_j L_j(H_j)^T psi_j + erreur
  #
  # En R, on le représente par :
  #
  #   Y^(j) ~ treatment-free terms + A_j * blip terms
  # --------------------------------------------------------------------

  .build_stage_formula = function(outcome_name, tf_formula, blip_matrix, treat_name) {

    # Termes du modèle treatment-free.
    tf_terms = attr(stats::terms(tf_formula), "term.labels")

    # Termes du blip sans l'intercept.
    blip_terms = setdiff(colnames(blip_matrix), "(Intercept)")

    # Construction du membre droit.
    rhs_tf = if (length(tf_terms) == 0L) {
      "1"
    } else {
      paste(tf_terms, collapse = " + ")
    }

    rhs_blip = if (length(blip_terms) == 0L) {
      "1"
    } else {
      paste(blip_terms, collapse = " + ")
    }

    stats::as.formula(
      paste0(
        outcome_name, " ~ ",
        rhs_tf, " + ",
        treat_name, " * (", rhs_blip, ")"
      )
    )
  }


  # --------------------------------------------------------------------
  # 8. Extraction des coefficients du blip
  # --------------------------------------------------------------------
  # Les paramètres psi_j correspondent :
  #   - au coefficient principal de A_j pour l'intercept du blip ;
  #   - aux coefficients d'interaction A_j:variable pour les autres termes.
  # --------------------------------------------------------------------

  .extract_blip_coefficients = function(model, treat_name, blip_names) {

    cf = stats::coef(model)

    if (anyNA(cf)) {
      .warn(
        "Des coefficients NA ont été détectés dans le modèle final. ",
        "Vérifie la colinéarité ou la positivité."
      )
    }

    out = rep(NA_real_, length(blip_names))
    names(out) = blip_names

    # Intercept du blip.
    if ("(Intercept)" %in% blip_names) {

      if (!(treat_name %in% names(cf))) {
        .stop(
          "Impossible d'extraire le coefficient principal de traitement ",
          treat_name,
          "."
        )
      }

      out["(Intercept)"] = unname(cf[treat_name])
    }

    # Autres termes du blip.
    other_names = setdiff(blip_names, "(Intercept)")

    for (nm in other_names) {

      # Deux notations possibles selon R :
      #   variable:traitement
      #   traitement:variable
      nm1 = paste0(nm, ":", treat_name)
      nm2 = paste0(treat_name, ":", nm)

      if (nm1 %in% names(cf)) {
        out[nm] = unname(cf[nm1])
      } else if (nm2 %in% names(cf)) {
        out[nm] = unname(cf[nm2])
      } else {
        .stop("Interaction introuvable pour ", nm, " et ", treat_name, ".")
      }
    }

    return(out)
  }


  # --------------------------------------------------------------------
  # 9. Construction sécurisée de la matrice du blip
  # --------------------------------------------------------------------

  .safe_model_matrix = function(form, dat, what) {

    mm = tryCatch(
      stats::model.matrix(form, data = dat),
      error = function(e) {
        .stop("Erreur dans ", what, " : ", e$message)
      }
    )

    if (!is.matrix(mm)) {
      .stop(what, " n'a pas produit une matrice valide.")
    }

    if (nrow(mm) != nrow(dat)) {
      .stop(what, " a un nombre de lignes incohérent.")
    }

    return(mm)
  }


  # --------------------------------------------------------------------
  # 10. Vérifications globales des arguments
  # --------------------------------------------------------------------

  if (!is.data.frame(data)) {
    .stop("data doit être un data.frame.")
  }

  if (nrow(data) == 0L) {
    .stop("data ne peut pas être vide.")
  }

  if (!.is_scalar_character(outcome)) {
    .stop("outcome doit être une chaîne de caractères de longueur 1.")
  }

  if (!(outcome %in% names(data))) {
    .stop("La variable outcome = '", outcome, "' est absente de data.")
  }

  if (!is.character(treat.names) || length(treat.names) == 0L) {
    .stop("treat.names doit être un vecteur de caractères non vide.")
  }

  if (anyNA(treat.names) || any(treat.names == "")) {
    .stop("treat.names contient une valeur manquante ou vide.")
  }

  if (anyDuplicated(treat.names)) {
    .stop("treat.names contient des doublons.")
  }

  # Nombre de temps de décision.
  K = length(treat.names)

  # Cette version est limitée à 1 ou 2 temps.
  if (!(K %in% c(1L, 2L))) {
    .stop("Cette version de dwols_sl gère uniquement 1 ou 2 temps.")
  }

  if (!is.list(H.vars) || length(H.vars) != K) {
    .stop("H.vars doit être une liste de longueur identique à length(treat.names).")
  }

  .check_formula_list(tf.formulas, "tf.formulas")
  .check_formula_list(blip.formulas, "blip.formulas")

  if (length(tf.formulas) != K) {
    .stop("tf.formulas doit avoir la même longueur que treat.names.")
  }

  if (length(blip.formulas) != K) {
    .stop("blip.formulas doit avoir la même longueur que treat.names.")
  }

  if (!.is_flag(use_cross_fitting)) {
    .stop("use_cross_fitting doit être TRUE ou FALSE.")
  }

  if (!is.numeric(folds) || length(folds) != 1L || is.na(folds) || folds < 2) {
    .stop("folds doit être un nombre >= 2.")
  }

  folds = as.integer(folds)

  if (!is.character(SL.library) || length(SL.library) == 0L) {
    .stop("SL.library doit être un vecteur de caractères non vide.")
  }

  if (!.is_scalar_character(SL.method)) {
    .stop("SL.method doit être une chaîne de caractères de longueur 1.")
  }

  if (!is.null(seed) && (!is.numeric(seed) || length(seed) != 1L || is.na(seed))) {
    .stop("seed doit être NULL ou un nombre unique.")
  }

  if (!.is_flag(return_fits)) {
    .stop("return_fits doit être TRUE ou FALSE.")
  }

  if (!.is_flag(verbose)) {
    .stop("verbose doit être TRUE ou FALSE.")
  }

  if (!.is_flag(bootstrap)) {
    .stop("bootstrap doit être TRUE ou FALSE.")
  }

  if (!is.numeric(B) || length(B) != 1L || is.na(B) || B < 2) {
    .stop("B doit être un entier >= 2.")
  }

  B = as.integer(B)

  if (!is.numeric(q) || length(q) != 1L || is.na(q) || q <= 0 || q >= 1) {
    .stop("q doit être un nombre strictement entre 0 et 1.")
  }

  if (!is.numeric(alpha) || length(alpha) != 1L || is.na(alpha) || alpha <= 0 || alpha >= 1) {
    .stop("alpha doit être un nombre strictement entre 0 et 1.")
  }

  if (!is.null(bootstrap_seed) &&
      (!is.numeric(bootstrap_seed) || length(bootstrap_seed) != 1L || is.na(bootstrap_seed))) {
    .stop("bootstrap_seed doit être NULL ou un nombre unique.")
  }


  # --------------------------------------------------------------------
  # 11. Vérifications sur les données
  # --------------------------------------------------------------------

  dat = data

  needed_vars = unique(c(
    outcome,
    treat.names,
    unlist(H.vars, use.names = FALSE)
  ))

  missing_needed = setdiff(needed_vars, names(dat))

  if (length(missing_needed) > 0L) {
    .stop(
      "Les variables suivantes sont absentes de data : ",
      paste(missing_needed, collapse = ", ")
    )
  }

  formula_vars = unique(c(
    unlist(lapply(tf.formulas, all.vars), use.names = FALSE),
    unlist(lapply(blip.formulas, all.vars), use.names = FALSE)
  ))

  missing_formula = setdiff(formula_vars, names(dat))

  if (length(missing_formula) > 0L) {
    .stop(
      "Les variables suivantes, utilisées dans les formules, sont absentes de data : ",
      paste(missing_formula, collapse = ", ")
    )
  }

  all_used_vars = unique(c(needed_vars, formula_vars))

  if (anyNA(dat[, all_used_vars, drop = FALSE])) {

    na_counts = colSums(is.na(dat[, all_used_vars, drop = FALSE]))
    na_counts = na_counts[na_counts > 0]

    .stop(
      "Des valeurs manquantes ont été détectées dans les variables utilisées : ",
      paste(names(na_counts), "(", na_counts, ")", collapse = ", ")
    )
  }

  if (!is.numeric(dat[[outcome]])) {
    .stop("L'outcome doit être numérique.")
  }

  if (any(!is.finite(dat[[outcome]]))) {
    .stop("L'outcome contient des valeurs non finies.")
  }

  # Vérifications des traitements.
  for (j in seq_len(K)) {

    Aj = dat[[treat.names[j]]]

    if (!is.numeric(Aj) && !is.integer(Aj)) {
      .stop(
        "Le traitement ",
        treat.names[j],
        " doit être numérique/integer et codé 0/1."
      )
    }

    if (any(!is.finite(Aj))) {
      .stop("Le traitement ", treat.names[j], " contient des valeurs non finies.")
    }

    uAj = sort(unique(Aj))

    if (!all(uAj %in% c(0, 1))) {
      .stop(
        "Le traitement ",
        treat.names[j],
        " doit être codé 0/1. Valeurs observées : ",
        paste(uAj, collapse = ", ")
      )
    }

    if (length(uAj) < 2L) {
      .stop(
        "Le traitement ",
        treat.names[j],
        " ne contient qu'une seule modalité dans les données."
      )
    }
  }

  # Vérifications des historiques.
  for (j in seq_len(K)) {

    if (!is.character(H.vars[[j]]) || length(H.vars[[j]]) == 0L) {
      .stop("H.vars[[", j, "]] doit être un vecteur de caractères non vide.")
    }

    if (anyNA(H.vars[[j]]) || any(H.vars[[j]] == "")) {
      .stop("H.vars[[", j, "]] contient une valeur manquante ou vide.")
    }

    if (anyDuplicated(H.vars[[j]])) {
      .stop("H.vars[[", j, "]] contient des doublons.")
    }

    if (!all(H.vars[[j]] %in% names(dat))) {
      .stop("Certaines variables de H.vars[[", j, "]] sont absentes de data.")
    }
  }

  if (use_cross_fitting && folds > nrow(dat)) {
    .stop("folds ne peut pas être supérieur au nombre d'observations.")
  }

  numeric_used = all_used_vars[
    vapply(dat[, all_used_vars, drop = FALSE], is.numeric, logical(1))
  ]

  if (length(numeric_used) > 0L) {

    bad_numeric = vapply(
      dat[, numeric_used, drop = FALSE],
      function(x) any(!is.finite(x)),
      logical(1)
    )

    if (any(bad_numeric)) {
      .stop(
        "Certaines variables numériques contiennent Inf/-Inf : ",
        paste(names(bad_numeric)[bad_numeric], collapse = ", ")
      )
    }
  }

  if (!is.null(seed)) {
    set.seed(as.integer(seed))
  }


  # --------------------------------------------------------------------
  # 12. Initialisation des objets de sortie
  # --------------------------------------------------------------------

  psi = vector("list", K)
  opt.treat = vector("list", K)
  propensity = vector("list", K)
  weights = vector("list", K)
  blip.design = vector("list", K)
  mu = vector("list", K)
  models = vector("list", K)
  ps_fits = vector("list", K)
  stage_formulas = vector("list", K)

  names(psi) = paste0("stage", seq_len(K))
  names(opt.treat) = paste0("stage", seq_len(K))
  names(propensity) = paste0("stage", seq_len(K))
  names(weights) = paste0("stage", seq_len(K))
  names(blip.design) = paste0("stage", seq_len(K))
  names(mu) = paste0("stage", seq_len(K))
  names(models) = paste0("stage", seq_len(K))
  names(ps_fits) = paste0("stage", seq_len(K))
  names(stage_formulas) = paste0("stage", seq_len(K))

  # Pseudo-outcome initial :
  #   Y^(K) = Y
  pseudoY = dat[[outcome]]


  # --------------------------------------------------------------------
  # 13. Récursion principale backward
  # --------------------------------------------------------------------
  # La fonction travaille du dernier temps vers le premier :
  #   j = K, K-1, ..., 1
  # --------------------------------------------------------------------

  if (verbose) {
    message("Début de dwols_sl()")
    message("Nombre de temps : ", K)
    message("Poids utilisés  : ", weight)
    message("Cross-fitting   : ", use_cross_fitting)
  }

  for (j in K:1) {

    if (verbose) {
      message("---- Temps ", j, " ----")
    }

    # Nom du traitement au temps j.
    Aj_name = treat.names[j]

    # Traitement observé au temps j.
    Aj = dat[[Aj_name]]

    # ------------------------------------------------------------------
    # 13.1 Construction de la matrice du blip
    # ------------------------------------------------------------------
    # On construit :
    #   L_j(H_j)
    # ------------------------------------------------------------------

    Lj = .safe_model_matrix(
      blip.formulas[[j]],
      dat,
      paste0("blip.formulas[[", j, "]]")
    )

    blip.design[[j]] = Lj


    # ------------------------------------------------------------------
    # 13.2 Estimation de la propension
    # ------------------------------------------------------------------
    # On estime :
    #   p_hat_j(H_j) = P_hat(A_j = 1 | H_j)
    # ------------------------------------------------------------------

    if (verbose) {
      message("Estimation de la propension pour ", Aj_name)
    }

    if (use_cross_fitting) {

      ps_obj = .crossfit_ps(
        A = Aj,
        H = dat[, H.vars[[j]], drop = FALSE],
        folds = folds,
        seed_local = if (is.null(seed)) NULL else as.integer(seed) + j
      )

      p_hat_j = ps_obj$p_hat

      if (return_fits) {
        ps_fits[[j]] = ps_obj
      }

    } else {

      fit_j = .fit_ps(
        A = Aj,
        H = dat[, H.vars[[j]], drop = FALSE]
      )

      p_hat_j = .predict_ps(
        fit = fit_j,
        newdata = dat[, H.vars[[j]], drop = FALSE]
      )

      if (return_fits) {
        ps_fits[[j]] = fit_j
      }
    }

    if (anyNA(p_hat_j)) {
      .stop("Des NA ont été produits dans les propensions estimées au temps ", j, ".")
    }

    if (any(!is.finite(p_hat_j))) {
      .stop("Des valeurs non finies ont été produites dans les propensions estimées au temps ", j, ".")
    }

    propensity[[j]] = p_hat_j


    # ------------------------------------------------------------------
    # 13.3 Construction des poids dWOLS
    # ------------------------------------------------------------------
    # Poids :
    #   w_j = |A_j - p_hat_j(H_j)|
    # ------------------------------------------------------------------

    w_j = .compute_weights(
      A = Aj,
      p_hat = p_hat_j
    )

    if (anyNA(w_j) || any(!is.finite(w_j))) {
      .stop("Les poids calculés au temps ", j, " contiennent des valeurs invalides.")
    }

    if (any(w_j < 0)) {
      .stop("Les poids calculés au temps ", j, " contiennent des valeurs négatives.")
    }

    weights[[j]] = w_j


    # ------------------------------------------------------------------
    # 13.4 Construction du pseudo-outcome
    # ------------------------------------------------------------------

    y_name = paste0(".pseudoY_stage_", j)
    dat[[y_name]] = pseudoY


    # ------------------------------------------------------------------
    # 13.5 Régression pondérée dWOLS
    # ------------------------------------------------------------------
    # On ajuste :
    #   Y^(j) ~ m_j(H_j) + A_j * L_j(H_j)
    # avec les poids w_j.
    # ------------------------------------------------------------------

    form_j = .build_stage_formula(
      outcome_name = y_name,
      tf_formula = tf.formulas[[j]],
      blip_matrix = Lj,
      treat_name = Aj_name
    )

    stage_formulas[[j]] = form_j

    if (verbose) {
      message("Estimation du modèle dWOLS au temps ", j)
    }

    fit_stage_j = tryCatch(
      stats::lm(
        form_j,
        data = dat,
        weights = w_j
      ),
      error = function(e) {
        .stop("Erreur dans le modèle dWOLS au temps ", j, " : ", e$message)
      }
    )

    if (return_fits) {
      models[[j]] = fit_stage_j
    }


    # ------------------------------------------------------------------
    # 13.6 Extraction de psi_hat_j
    # ------------------------------------------------------------------
    # psi_hat_j est extrait de la partie blip :
    #   A_j x L_j(H_j)
    # ------------------------------------------------------------------

    psi_j = .extract_blip_coefficients(
      model = fit_stage_j,
      treat_name = Aj_name,
      blip_names = colnames(Lj)
    )

    psi[[j]] = psi_j


    # ------------------------------------------------------------------
    # 13.7 Règle optimale estimée
    # ------------------------------------------------------------------
    # Score de décision :
    #   L_j(H_j)^T psi_hat_j
    #
    # Règle :
    #   d_hat_j(H_j) = 1{L_j(H_j)^T psi_hat_j > 0}
    # ------------------------------------------------------------------

    linpred_opt = drop(Lj %*% psi_j)

    if (anyNA(linpred_opt) || any(!is.finite(linpred_opt))) {
      .stop("Le score de décision au temps ", j, " contient des valeurs invalides.")
    }

    dopt_j = as.numeric(linpred_opt > 0)
    opt.treat[[j]] = dopt_j


    # ------------------------------------------------------------------
    # 13.8 Contribution au regret
    # ------------------------------------------------------------------
    # Contribution stage-specific :
    #   mu_j = (d_hat_j(H_j) - A_j) L_j(H_j)^T psi_hat_j
    # ------------------------------------------------------------------

    mu_j = (dopt_j - Aj) * linpred_opt

    if (anyNA(mu_j) || any(!is.finite(mu_j))) {
      .stop("La contribution mu au temps ", j, " contient des valeurs invalides.")
    }

    mu[[j]] = mu_j


    # ------------------------------------------------------------------
    # 13.9 Mise à jour backward du pseudo-outcome
    # ------------------------------------------------------------------
    # Mise à jour :
    #   Y^(j-1) = Y^(j) + mu_j
    # ------------------------------------------------------------------

    pseudoY = pseudoY + mu_j
  }


  # --------------------------------------------------------------------
  # 14. Quantités finales
  # --------------------------------------------------------------------

  # Regret total :
  #   regret = sum_j mu_j
  regret = Reduce(`+`, mu)

  # Outcome optimal estimé :
  #   Y_opt = Y + regret
  opt.Y = dat[[outcome]] + regret

  if (anyNA(regret) || any(!is.finite(regret))) {
    .stop("Le regret final contient des valeurs invalides.")
  }

  if (anyNA(opt.Y) || any(!is.finite(opt.Y))) {
    .stop("opt.Y contient des valeurs invalides.")
  }

  # Concaténation des psi dans un seul vecteur.
  psi.vector = unlist(psi, use.names = TRUE)


  # --------------------------------------------------------------------
  # 15. m-out-of-n bootstrap pour les intervalles de confiance
  # --------------------------------------------------------------------
  # Cette fonction interne calcule les intervalles de confiance pour
  # les paramètres du blip à partir d'un bootstrap m-out-of-n.
  #
  # Étapes :
  #   1. Construire une grille de tailles m_j = floor(q^j n).
  #   2. Pour chaque m_j, générer B échantillons bootstrap de taille m_j.
  #   3. Réestimer dwols_sl sur chaque échantillon bootstrap.
  #   4. Comparer les distributions successives pour choisir m optimal.
  #   5. Construire les intervalles de confiance.
  # --------------------------------------------------------------------

  .m_out_of_n_bootstrap = function() {

    # Taille de l'échantillon original.
    n = nrow(data)

    # Estimateur sur l'échantillon original.
    psi.n = psi.vector

    # Nombre total de paramètres blip.
    P = length(psi.n)

    # Nombre de tailles bootstrap considérées.
    Kboot = ceiling(log(0.5) / log(q))

    # Grille des tailles bootstrap :
    #   m_j = floor(q^j n)
    m_grid = floor(q^(0:Kboot) * n)

    # On élimine par sécurité les tailles trop petites pour le cross-fitting.
    m_grid = m_grid[m_grid >= max(2, folds)]

    if (length(m_grid) < 2) {
      .stop("La grille m_grid contient moins de deux tailles valides.")
    }

    # Liste des estimateurs bootstrap.
    psi.boot = vector("list", length(m_grid))

    # Variances bootstrap.
    V = matrix(
      NA_real_,
      nrow = length(m_grid),
      ncol = P
    )

    if (!is.null(bootstrap_seed)) {
      set.seed(as.integer(bootstrap_seed))
    }

    # ------------------------------------------------------------------
    # 15.1 Boucle sur les tailles bootstrap m_j
    # ------------------------------------------------------------------

    for (jj in seq_along(m_grid)) {

      m = m_grid[jj]

      if (verbose) {
        message(
          "Bootstrap m-out-of-n : m = ",
          m,
          " (",
          jj,
          "/",
          length(m_grid),
          ")"
        )
      }

      # Matrice des estimateurs bootstrap :
      # lignes = réplications bootstrap
      # colonnes = paramètres psi
      boot_mat = matrix(
        NA_real_,
        nrow = B,
        ncol = P
      )

      # --------------------------------------------------------------
      # 15.2 Boucle bootstrap
      # --------------------------------------------------------------

      for (b in seq_len(B)) {

        # Tirage bootstrap avec remise de taille m.
        idx = sample(
          seq_len(n),
          size = m,
          replace = TRUE
        )

        dat_b = data[idx, , drop = FALSE]

        # Ajustement de dwols_sl sur l'échantillon bootstrap.
        #
        # Important :
        #   bootstrap = FALSE
        # évite une récursion infinie.
        fit_b = tryCatch(
          dwols_sl(
            data = dat_b,
            outcome = outcome,
            treat.names = treat.names,
            H.vars = H.vars,
            tf.formulas = tf.formulas,
            blip.formulas = blip.formulas,
            weight = "default",
            use_cross_fitting = use_cross_fitting,
            folds = min(folds, m),
            SL.library = SL.library,
            SL.method = SL.method,
            seed = NULL,
            return_fits = FALSE,
            verbose = FALSE,
            bootstrap = FALSE
          ),
          error = function(e) NULL
        )

        if (!is.null(fit_b)) {
          boot_mat[b, ] = fit_b$psi.vector
        }
      }

      colnames(boot_mat) = names(psi.n)

      # Sauvegarde des estimateurs bootstrap.
      psi.boot[[jj]] = boot_mat

      # Variance bootstrap empirique.
      V[jj, ] = apply(
        boot_mat,
        2,
        stats::var,
        na.rm = TRUE
      )
    }


    # ------------------------------------------------------------------
    # 15.3 Bornes des grilles empiriques
    # ------------------------------------------------------------------

    x_min = rep(Inf, P)
    x_max = rep(-Inf, P)

    for (jj in seq_along(m_grid)) {

      # Quantité centrée et normalisée :
      #   sqrt(m_j) (psi_m - psi_n)
      diff_j =
        sqrt(m_grid[jj]) *
        sweep(
          psi.boot[[jj]],
          2,
          psi.n,
          "-"
        )

      x_min = pmin(
        x_min,
        apply(diff_j, 2, min, na.rm = TRUE)
      )

      x_max = pmax(
        x_max,
        apply(diff_j, 2, max, na.rm = TRUE)
      )
    }


    # ------------------------------------------------------------------
    # 15.4 Sélection optimale de m
    # ------------------------------------------------------------------
    # Pour chaque paramètre, on compare les fonctions de répartition
    # empiriques associées à deux tailles bootstrap successives.
    # On retient le m qui minimise la distance maximale.
    # ------------------------------------------------------------------

    supR = matrix(
      NA_real_,
      nrow = length(m_grid) - 1,
      ncol = P
    )

    m_opt = numeric(P)
    V_opt = numeric(P)
    beta = numeric(P)

    for (p in seq_len(P)) {

      x_grid = seq(
        x_min[p],
        x_max[p],
        length.out = 100
      )

      for (jj in seq_len(length(m_grid) - 1)) {

        R_j = numeric(length(x_grid))

        for (xx in seq_along(x_grid)) {

          R_j[xx] =
            mean(
              sqrt(m_grid[jj]) *
                (psi.boot[[jj]][, p] - psi.n[p]) <= x_grid[xx],
              na.rm = TRUE
            ) -
            mean(
              sqrt(m_grid[jj + 1]) *
                (psi.boot[[jj + 1]][, p] - psi.n[p]) <= x_grid[xx],
              na.rm = TRUE
            )
        }

        # Distance maximale entre deux fonctions de répartition empiriques.
        supR[jj, p] = max(
          abs(R_j),
          na.rm = TRUE
        )
      }

      # Indice optimal.
      jopt = which.min(supR[, p])

      # Taille bootstrap optimale.
      m_opt[p] = m_grid[jopt]

      # Variance bootstrap optimale.
      V_opt[p] = V[jopt, p]


      # --------------------------------------------------------------
      # 15.5 Estimation de beta
      # --------------------------------------------------------------

      logVm = log(V[, p])
      logm = -2 * log(m_grid)

      ok = is.finite(logVm) & is.finite(logm)

      if (sum(ok) >= 2) {
        beta[p] = stats::coef(
          stats::lm(logVm[ok] ~ logm[ok])
        )[2]
      } else {
        beta[p] = NA_real_
      }
    }


    # ------------------------------------------------------------------
    # 15.6 Construction des intervalles de confiance
    # --------------------------------------------------------------------

    z = stats::qnorm(1 - alpha / 2)

    lower =
      psi.n -
      z *
      (m_opt / n)^beta *
      sqrt(V_opt)

    upper =
      psi.n +
      z *
      (m_opt / n)^beta *
      sqrt(V_opt)

    names(lower) = names(psi.n)
    names(upper) = names(psi.n)
    names(m_opt) = names(psi.n)
    names(beta) = names(psi.n)
    names(V_opt) = names(psi.n)

    ci = data.frame(
      parameter = names(psi.n),
      estimate = as.numeric(psi.n),
      lower = as.numeric(lower),
      upper = as.numeric(upper),
      m_opt = as.numeric(m_opt),
      variance = as.numeric(V_opt),
      beta = as.numeric(beta),
      row.names = NULL
    )

    list(
      method = "m-out-of-n bootstrap",
      B = B,
      q = q,
      alpha = alpha,
      m_grid = m_grid,
      m_opt = m_opt,
      beta = beta,
      variance = V_opt,
      lower = lower,
      upper = upper,
      ci = ci,
      psi.boot = psi.boot,
      supR = supR
    )
  }


  # --------------------------------------------------------------------
  # 16. Construction de la sortie
  # --------------------------------------------------------------------

  out = list(
    call = match.call(),
    K = K,
    outcome = outcome,
    treat.names = treat.names,
    psi = psi,
    psi.vector = psi.vector,
    opt.treat = opt.treat,
    propensity = propensity,
    weights = weights,
    mu = mu,
    regret = regret,
    opt.Y = opt.Y,
    blip.design = blip.design,
    stage.formulas = stage_formulas,
    settings = list(
      weight = weight,
      use_cross_fitting = use_cross_fitting,
      folds = folds,
      SL.library = SL.library,
      SL.method = SL.method,
      seed = seed,
      bootstrap = bootstrap,
      B = B,
      q = q,
      alpha = alpha,
      bootstrap_seed = bootstrap_seed
    )
  )

  if (return_fits) {
    out$models = models
    out$ps_fits = ps_fits
  }

  # Calcul optionnel des IC par m-out-of-n bootstrap.
  if (bootstrap) {

    if (verbose) {
      message("Calcul des intervalles de confiance par m-out-of-n bootstrap...")
    }

    out$bootstrap = .m_out_of_n_bootstrap()
  }

  class(out) = "dwols_sl"

  return(out)
}


