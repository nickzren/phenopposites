# Download the PATO ontology
wget --timestamping --directory-prefix data/input/ https://raw.githubusercontent.com/pato-ontology/pato/master/pato-full.owl

# Download the HPO ontology
wget --timestamping --directory-prefix data/input/ https://purl.obolibrary.org/obo/hp.obo
wget --timestamping --directory-prefix data/input/ https://purl.obolibrary.org/obo/hp.owl

# Download the MONDO ontology
wget --timestamping --directory-prefix data/input/ http://purl.obolibrary.org/obo/mondo.obo