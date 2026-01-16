import { formatLinkLabel } from '../lib/chat-helpers';
import { BotsIconMini } from './icons';

interface LinkCardProps {
  href: string;
}

export const LinkCard = ({ href }: LinkCardProps) => (
  <a className="link-card" href={href} target="_blank" rel="noreferrer">
    <div className="link-card__icon">
      <BotsIconMini />
    </div>
    <div className="link-card__body">
      <p className="link-card__title">{formatLinkLabel(href)}</p>
      <span className="link-card__url">{href}</span>
    </div>
  </a>
);
