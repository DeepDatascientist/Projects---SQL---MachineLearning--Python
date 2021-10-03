#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import phonenumbers
from phonenumbers import carrier
from phonenumbers import geocoder

a = input ("Enter Your Phone Number :")
phoen_number = phonenumbers.parse(a)

print(geocoder.description_for_number(phone_number,"en"))
print(carrier.name_for_number(phone_number,"en"))


# In[ ]:




