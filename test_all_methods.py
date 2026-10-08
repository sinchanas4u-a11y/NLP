"""Test all three methods with detailed calculations"""

from app import display_all_methods_calculations  # type: ignore

text = 'Deep learning models are used for disease prediction and drug discovery.'

print('='*70)
print('INPUT TEXT')
print('='*70)
print(text)
print()

display_all_methods_calculations(text)
