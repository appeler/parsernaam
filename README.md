## Parsernaam: Predict First or Last Name

Most common name parsers use crude pattern matching and the sequence of strings, e.g., the last word is the last name, to parse names. This approach is limited and fragile, especially for Indian names. We take a machine-learning approach to the problem. Using the large voter registration data in India and US, we build machine-learning-based name parsers that predict whether the string is a first or last name. 

For Indian electoral rolls, we assume the last name is the word in the name that is shared by multiple family members. (We table the expansion to include compound last names---extremely rare in India---till the next iteration.)

### Authors
Rajashekar Chintalapati and Gaurav Sood

