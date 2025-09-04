from llama_index.core import PromptTemplate  

instruction_str = """\
1. If the query requests information about a single person (e.g., 'give me details of John Doe'), write a single Python expression using Pandas to answer the query and return a dictionary or JSON-like structure containing the complete information of that person.
2. If the query asks for a list of people based on specific criteria (e.g., 'give a list of people from New York'), write a single Python expression that returns a list of dictionaries, with each dictionary containing the complete information of each person who matches the criteria.
3. Ensure the code can be executed with `eval()` by keeping it as a single expression.
4. The expression should ignore case sensitivity when checking any column and data values in the column.
5. Only output the final expression without quotation marks.
6. The columns in the DataFrame are: 'timestamp', 'id', 'name', 'city', 'country_code', 'region', 'current_company:company_id', 'current_company:name', 'position', 'following', 'about', 'posts', 'groups', 'current_company', 'experience', 'url', 'people_also_viewed', 'educations_details', 'education', 'avatar', 'languages', 'certifications', 'recommendations', 'recommendations_count', 'volunteer_experience', 'courses'.
7.In the dataframe, column 'city' refers to the country
8.In the dataframe, column 'current_company:name' refers to the company.
9.In the dataframe, column 'about' refers to the about the person.
10.In the dataframe, column 'experience' refers to the experience.
11.In the dataframe, column 'url' refers to the linkeldn profile.
12.In the dataframe, column 'education' refers to the qualifications.
13.In the dataframe, column 'educations_details' refers to the education information.
14.In the dataframe, column 'languages' refers to the language proficiency or languages known.
15.In the dataframe, column 'certifications' refers to the skill or certifications.
16.In the dataframe, column 'recommendations_count' refers to the recommendation.
17.Give the name of person when the recommendation value matches.
"""  


new_prompt = PromptTemplate(  
    """\
    You are working with a pandas DataFrame in Python.  
    The name of the DataFrame is df.  
    This is the result of print(df.head()):  
    {df_str}  

    Follow these instructions:  
    {instruction_str}  

    Query:  
    {query_str}  

    Expression:  
    """  
)  

context = """  
Purpose: The primary role of this agent is to assist users by providing accurate and 
comprehensive information about LinkedIn profiles of people. 
The agent should be able to handle complex queries related to any of the fields in the 'population.csv' file 
and return the complete information of the person or a list of people as dictionaries or JSON-like structures 
based on the query criteria.
"""    
