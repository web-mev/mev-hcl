process run_hcl {

    tag "HCL process"
    publishDir "${params.output_dir}/HierarchicalCluster.observation_clusters", mode:"copy", pattern:"${obs_file}"
    publishDir "${params.output_dir}/HierarchicalCluster.feature_clusters", mode:"copy", pattern:"${feature_file}"
    container "ghcr.io/web-mev/mev-hcl"
    cpus 4
    memory '32 GB'

    input:
        path(input_file)

    output:
        path("${obs_file}")
        path("${feature_file}")
    script:
        obs_file = "observations.tsv"
        feature_file = "features.tsv"
        """
        echo "{}" > ${obs_file}
        echo "{}" > ${feature_file}
        python3 /usr/local/bin/run_hcl.py -i ${input_file} \
            -d ${params.distance_metric} \
            -l ${params.linkage} \
            -c ${params.clustering_dimension} \
            -o ${obs_file} \
            -f ${feature_file}
        """
}

workflow {

    run_hcl(params.input_matrix)

}
