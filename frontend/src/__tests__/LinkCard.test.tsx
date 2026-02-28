import { render, screen } from '@testing-library/react';

import { LinkCard } from '../app/components/LinkCard';

describe('LinkCard', () => {
  it('renders normalized hostname and href', () => {
    render(<LinkCard href="https://www.example.com/path?q=1" />);

    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', 'https://www.example.com/path?q=1');
    expect(screen.getByText('example.com')).toBeInTheDocument();
    expect(screen.getByText('https://www.example.com/path?q=1')).toBeInTheDocument();
  });
});
