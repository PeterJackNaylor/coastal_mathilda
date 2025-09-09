
keywords = ["daily_beach_spatial", "daily_beach_spatial_encoding", "daily_beach_temporal", "daily_beach_temporal_encoding",
            "monthly_beach_spatial", "monthly_beach_spatial_encoding", "monthly_beach_temporal", "monthly_beach_temporal_encoding",
            "weekly_beach_spatial", "weekly_beach_spatial_encoding", "weekly_beach_temporal", "weekly_beach_temporal_encoding",
            "seasonal_beach_spatial", "seasonal_beach_spatial_encoding", "seasonal_beach_temporal", "seasonal_beach_temporal_encoding"]

data = file("./data/")
pyinr = file("src/single_run.py")
yaml_file = file("mathilda.yml")

process INR {
    publishDir "outputs"
    queue 'gpu4_std'
    clusterOptions "-l select=1:ncpus=8:ngpus=1:mem=80gb -l walltime=24:00:00"
    input:
        path pyinr
        path data
        path yaml
        each key

    output:
        path("${key}")

    script:
        """
            python $pyinr --name ${key} --keyword $key     
        """
}

pyaggregate = file("src/aggregate.py")

process aggregate {
    publishDir "outputs"
    input:
        path result_files
        path pyagg
    output:
        path "results.csv"

    script:
        """
            python $pyagg
        """
}




workflow {

    main:
        test = INR(pyinr, data, yaml_file, keywords)
        
        aggregate(test.collect(), pyaggregate)
    }
