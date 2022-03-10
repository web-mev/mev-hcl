workflow HierarchicalCluster {

    # The input matrix
    File input_matrix

    # Parameters for the actual clustering
    String distance_metric
    String clustering_dimension
    String linkage

    call runCluster {
        input:
            input_matrix = input_matrix,
            distance_metric = distance_metric,
            clustering_dimension = clustering_dimension,
            linkage = linkage
    }

    output {
        File? observation_clusters = runCluster.observation_clusters
        File? feature_clusters = runCluster.feature_clusters
    }

}

task runCluster {
        
    # The input matrix
    File input_matrix

    # Parameters for the actual clustering
    String distance_metric
    String clustering_dimension
    String linkage

    String observations_output = "observations_hcl.json"
    String features_output = "features_hcl.json"
    Int disk_size = 50

    command <<<
        python3 /opt/software/run_hcl.py -i ${input_matrix} \
            -d ${distance_metric} \
            -l ${linkage} \
            -c ${clustering_dimension} \
            -o ${observations_output} \
            -f ${features_output}
    >>>

    output {
        File? observation_clusters = "${observations_output}"
        File? feature_clusters = "${features_output}"
    }

    runtime {
        docker: "ghcr.io/web-mev/mev-hcl"
        cpu: 4
        memory: "30 G"
        disks: "local-disk " + disk_size + " HDD"
        preemptible: 0
    }
}



