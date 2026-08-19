#!/bin/bash
set -e

CLOUD=$1
shift

ONLY_SUBMODULE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only-submodule) ONLY_SUBMODULE="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"

if [[ -f "module.env" ]]; then
    EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
    EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE"
fi

if [[ -f "module_${CLOUD}.env" ]]; then
    EXTRA_PARAMS_MODULE_CLOUD=$(paste -sd, "module_${CLOUD}.env")
    EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE_CLOUD"
fi

echo "Extra Params: $EXTRA_PARAMS"

# Resolve the UI app's service-principal client id so app-dispatched jobs can grant CAN_MANAGE_RUN
# DECLARATIVELY (see app_sp_client_id in pca_v1 variables.yml). A job's permissions: block is its
# complete desired ACL — without this, every bundle deploy wipes the grant until initialize.py
# re-adds it. Best effort: if the app does not exist yet the var stays empty.
APP_NAME_FIRST=$(printf '%s' "$EXTRA_PARAMS" | tr ',' '\n' | grep '^databricks_app_names=' | head -1 | cut -d= -f2- | cut -d: -f1)
if [[ -n "$APP_NAME_FIRST" ]]; then
    APP_SP_CLIENT_ID=$(databricks apps get "$APP_NAME_FIRST" --output json 2>/dev/null | jq -r '.service_principal_client_id // empty')
    if [[ -n "$APP_SP_CLIENT_ID" ]]; then
        EXTRA_PARAMS="$EXTRA_PARAMS,app_sp_client_id=$APP_SP_CLIENT_ID"
        echo "Resolved app SP for '$APP_NAME_FIRST': $APP_SP_CLIENT_ID (granted CAN_MANAGE_RUN declaratively)"
    else
        echo "⚠️  Could not resolve a service principal for app '$APP_NAME_FIRST' — app-dispatched jobs"
        echo "    will rely on initialize.py's grant instead. Expected on a first install (no app yet)."
    fi
fi

echo "Extra Params: $EXTRA_PARAMS"

echo "##############################################"
echo "⏩️ Starting deploy of Genomics module  #"

ALL_SUBMODULES=(gwas/gwas_v1 vcf_ingestion/vcf_ingestion_v1 variant_annotation/variant_annotation_v1 parabricks/parabricks_v1 pca/pca_v1 prs/prs_v1)

if [[ -n "$ONLY_SUBMODULE" ]]; then
    found=false
    for s in "${ALL_SUBMODULES[@]}"; do
        if [[ "$s" == "$ONLY_SUBMODULE" ]]; then found=true; break; fi
    done
    if [[ "$found" != "true" ]]; then
        echo "Error: --only-submodule must be one of: ${ALL_SUBMODULES[*]}"
        exit 1
    fi
    SUBMODULES=("$ONLY_SUBMODULE")
else
    SUBMODULES=("${ALL_SUBMODULES[@]}")
fi

for module in "${SUBMODULES[@]}"
    do
        echo "###########################################"
        echo "Deploying $module"
        cd $module
        chmod +x deploy.sh

        echo "Running command deploy.sh $CLOUD --var=\"$EXTRA_PARAMS\" "
        ./deploy.sh $CLOUD --var="$EXTRA_PARAMS"
        cd ../..
    done
echo "##############################################"

if [[ -z "$ONLY_SUBMODULE" ]]; then
    date +"%Y-%m-%d %H:%M:%S" > .deployed
fi
