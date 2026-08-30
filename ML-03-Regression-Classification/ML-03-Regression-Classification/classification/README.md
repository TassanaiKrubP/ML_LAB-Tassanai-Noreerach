# Classification: Attacker vs Midfielder

Separates attackers (ST, CF, LW, RW) from midfielders (CM, CDM, CAM, LM, RM)
using the same 28 skill attributes.

Attacker vs Defender was tried first but scored 99.5%, which leaves nothing
to analyse. Attackers and midfielders genuinely overlap, so the confusion
matrix and ROC curve show real behaviour.

Pipeline: StandardScaler → PCA (10 components) → LogisticRegression.

Step 7 retrains on `finishing` and `long_passing` only, because a decision
boundary can only be drawn on two axes.
