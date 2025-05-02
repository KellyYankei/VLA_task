# VLA_maniskill_MedicineSort

## Description
The robot needs to categorize medical items (pills, box-pack medicines, and syringes) into their corresponding boxes.

## Objects
- **Table**: Worksurface where items are placed
- **Boxes (for categorization)**: Destination containers for sorted items
- **Pills**: Small round medicines to be sorted
- **Box-pack medicines**: Packaged medicines to be sorted
- **Cylinder-like syringes**: Medical syringes to be sorted

## Atomic Actions
- `Pick`: Grasp an object
- `Place`: Put an object into a container

## Reasoning Process
1. **Locate objects**: Identify all medicine-objects that need categorization
2. **Categorize items**: Determine the category of each medicine-objects
3. **Locate boxes**: Find all destination boxes
4. **Identify box categories**: Determine which box corresponds to which medicine type
5. **Sort items**: Place each medicine into its correct box until all are categorized

## Sim-to-Real Gap Considerations
- Need to label both the boxes and medicines for proper identification
- Potential challenges in object recognition and grasping

## Update Log (2025/5/2)
The basic environment is set. Potential improvements:
1. **More actors**: Add additional medicine types/variations
2. **Box labeling**: Implement clear labeling system for boxes
3. **Reward function**: Enhance the rewarding mechanism for better training


