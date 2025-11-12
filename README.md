# Set Master

This project is about the game of set, a card game where players aim to identify a "set" of three cards based on their shapes, colors, numbers, and shading.

## Features

- Synthetic dataset generation for training and testing
- Implementation of algorithms to identify sets
- Visualization tools to display cards and sets

## Background

### Game Rules

The game consists of a deck of 81 unique cards. Each card has four features:
1. Shape: Oval, Squiggle, or Diamond
2. Color: Red, Green, or Purple
3. Number: One, Two, or Three shapes on the card
4. Shading: Solid, Striped, or Open

A "set" consists of three cards in which, for each feature, the values are either all the same or all different.
For example, three cards that are all red, have different shapes, and different numbers would form a set.

### How to Play
1. The game starts with 12 cards laid out on the table.
2. Players take turns identifying sets among the cards on the table.
3. When a player identifies a set, they call it out and point to the three cards.
4. If the set is valid, the player collects the cards, and three new cards are dealt from the deck.
5. If no sets are found among the 12 cards, three additional cards are dealt.
6. The game continues until the deck is exhausted and no more sets can be found on the table.