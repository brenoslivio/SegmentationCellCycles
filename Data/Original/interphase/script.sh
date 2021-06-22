!#/bin/bash
for filename in *.json
do
  labelme_json_to_dataset "$filename" -o "${filename%.json}"
done


