
number=1


# Create a CSV file and write the header
echo "Domain, FID" > fid_results_odoc.csv

while [ "$number" -lt 6 ]; do
    number2=1
    while [ "$number2" -lt 6 ]; do

    # Run the Python command and capture the fid value
        fid_value=$(python -m pytorch_fid ../data/ODOC/Domain$number/train/imgs ../data/ODOC/Domain$number2/train/imgs --device cuda:0 --batch-size 30)

        # Append the fid value to the CSV file
        echo "Domain$number-Domain$number2, $fid_value" >> fid_results_odoc.csv
        number2=$((number2 + 1))
    done

    number=$((number + 1))
    
done