import cohere
co = cohere.Client()

paragraph = """
Giving gifts should always be enjoyable.  
  However, it may become stressful when trying to find that perfect present.   
  This wikiHow will help you figure out exactly what you'd love to receive this year!   
  If you're having trouble deciding between two different items (or more), 
  try making lists of advantages\/disadvantages so you'll know which one would make the best choice.    
  Make sure it's appropriate - some people don't appreciate receiving certain types of presents from their friends and\/or family members.     
  [MASK]
  these might have been forgotten by others who haven't seen them yet.        
  Write down all... Continue reading \u2192\n\nIf you can't think of anything specific right away but still feel like getting something nice,  
  consider giving yourself a treat instead!  
  Here are just a few suggestions:  
  Buying yourself flowers Gifting yourself tickets to see your favorite band Take time off work Treat yourself to lunch or dinner at your favorite restaurant 
  Spend extra money on your wardrobe Have a manicure Give yourself a massage Book a hotel room for the night"""
masked = " Don't forget to include any special requests you've already made before now;"

prompt = f"""
Replace [MASK] in following: {paragraph} with one sentence that has a meaning similar to: {masked}.
"""
'''
response = co.generate(
  prompt
)

print(response)
'''
print(prompt)