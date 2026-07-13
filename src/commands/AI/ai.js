import OpenAI from 'openai';
import { SlashCommandBuilder, MessageFlags, AttachmentBuilder } from 'discord.js';
import { createEmbed } from '../../utils/embeds.js';
import { InteractionHelper } from '../../utils/interactionHelper.js';
import { handleInteractionError, TitanBotError, ErrorTypes } from '../../utils/errorHandler.js';
import { logger } from '../../utils/logger.js';
import { ModerationService } from '../../services/moderationService.js';

const openai = new OpenAI({
    apiKey: process.env.NVIDIA_API_KEY || 'enter-your-api-key-here',
    baseURL: process.env.NVIDIA_BASE_URL || 'https://integrate.api.nvidia.com/v1',
});

const CHAT_MODEL = process.env.NVIDIA_CHAT_MODEL || 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning';
const IMAGE_MODEL = process.env.NVIDIA_IMAGE_MODEL || 'stabilityai/stable-diffusion-xl-base-1.0';

const SYSTEM_PROMPT = `
You are a helpful AI assistant and server moderation assistant.

When the user asks you to perform a moderation action (ban, kick, mute/timeout, unban, remove timeout), call the appropriate tool.
If the user provides a username or display name instead of an ID, tell them you need the numeric Discord user ID to proceed.
If a moderation action fails due to permissions or hierarchy, explain clearly what went wrong and tell the user to check if the target has a role with Administrator permission or a role higher than the bot's role in Server Settings → Roles.
If no reason is provided for a moderation action, use "No reason provided" as the reason and proceed without asking.
When performing moderation actions, the user_id field accepts a Discord user ID, a mention like @username, or a display name/username to search for. Use whatever the moderator provides.


Output rules for normal plain text responses:
- Produce clean, readable plain text.
- Never use Markdown headings (#, ##, ###).
- Never use horizontal rules (---).
- Use paragraphs and simple bullet lists (-) only when appropriate.
- Use fenced code blocks only for code or terminal commands.
`.trim();

const imageGenEnabled = new Map();
function isImageGenEnabled(guildId) {
    return imageGenEnabled.get(guildId) !== false;
}

function isBotOwner(userId) {
    const owners = (process.env.OWNER_IDS || '').split(',').map(id => id.trim()).filter(Boolean);
    return owners.includes(userId);
}

const MODERATION_TOOLS = [
    {
        type: 'function',
        function: {
            name: 'ban_user',
            description: 'Ban a user from the server, optionally deleting their recent messages.',
            parameters: {
                type: 'object',
                properties: {
                    user_id: { type: 'string', description: 'The Discord user ID to ban.' },
                    reason: { type: 'string', description: 'Reason for the ban.' },
                    delete_days: {
                        type: 'integer',
                        description: 'Number of days of messages to delete (0–7). Default 0.',
                        minimum: 0,
                        maximum: 7,
                    },
                },
                required: ['user_id', 'reason'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'kick_user',
            description: 'Kick a member from the server.',
            parameters: {
                type: 'object',
                properties: {
                    user_id: { type: 'string', description: 'The Discord user ID to kick.' },
                    reason: { type: 'string', description: 'Reason for the kick.' },
                },
                required: ['user_id', 'reason'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'timeout_user',
            description: 'Temporarily mute a member (timeout/mute).',
            parameters: {
                type: 'object',
                properties: {
                    user_id: { type: 'string', description: 'The Discord user ID to timeout.' },
                    duration_minutes: { type: 'integer', description: 'Timeout duration in minutes (1–40320).', minimum: 1, maximum: 40320 },
                    reason: { type: 'string', description: 'Reason for the timeout.' },
                },
                required: ['user_id', 'duration_minutes', 'reason'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'remove_timeout',
            description: 'Remove a timeout from a member early.',
            parameters: {
                type: 'object',
                properties: {
                    user_id: { type: 'string', description: 'The Discord user ID to un-timeout.' },
                    reason: { type: 'string', description: 'Reason for removing the timeout.' },
                },
                required: ['user_id', 'reason'],
            },
        },
    },
    {
        type: 'function',
        function: {
            name: 'unban_user',
            description: 'Unban a previously banned user.',
            parameters: {
                type: 'object',
                properties: {
                    user_id: { type: 'string', description: 'The Discord user ID to unban.' },
                    reason: { type: 'string', description: 'Reason for the unban.' },
                },
                required: ['user_id', 'reason'],
            },
        },
    },
];

async function resolveUser(input, guild, client) {
    const mentionMatch = input.match(/^<@!?(\d+)>$/);
    if (mentionMatch) input = mentionMatch[1];

    if (/^\d+$/.test(input)) {
        const user = await client.users.fetch(input).catch(() => null);
        const member = await guild.members.fetch(input).catch(() => null);
        return { user: user || member?.user || null, member: member || null };
    }

    const results = await guild.members.search({ query: input, limit: 5 }).catch(() => null);
    if (results?.size) {
        const lower = input.toLowerCase();
        const exact = results.find(
            m => m.user.username.toLowerCase() === lower ||
                 m.displayName.toLowerCase() === lower
        );
        const member = exact || results.first();
        return { user: member.user, member };
    }

    return { user: null, member: null };
}

async function executeTool(toolName, args, { guild, moderatorMember, client }) {
    const reason = args.reason || 'No reason provided';
    const { user: resolvedUser, member: resolvedMember } = await resolveUser(args.user_id, guild, client);

    switch (toolName) {
        case 'ban_user': {
            if (!resolvedUser) return { success: false, error: `Could not find user "${args.user_id}".` };
            try {
                await ModerationService.banUser({
                    guild,
                    user: resolvedUser,
                    moderator: moderatorMember,
                    reason,
                    deleteDays: args.delete_days ?? 0,
                });
            } catch (err) {
                return { success: false, error: `Cannot ban this member. They may have Administrator or a role above mine in Server Settings → Roles.` };
            }
            return { success: true, action: 'ban', user: resolvedUser.tag, reason, delete_days: args.delete_days ?? 0 };
        }

        case 'kick_user': {
            if (!resolvedMember) return { success: false, error: `Could not find member "${args.user_id}" in this server.` };
            try {
                await ModerationService.kickUser({ guild, member: resolvedMember, moderator: moderatorMember, reason });
            } catch (err) {
                return { success: false, error: `Cannot kick this member. They may have Administrator or a role above mine.` };
            }
            return { success: true, action: 'kick', user: resolvedMember.user.tag, reason };
        }

        case 'timeout_user': {
            if (!resolvedMember) return { success: false, error: `Could not find member "${args.user_id}" in this server.` };
            const durationMs = (args.duration_minutes ?? 5) * 60 * 1000;
            try {
                await ModerationService.timeoutUser({ guild, member: resolvedMember, moderator: moderatorMember, durationMs, reason });
            } catch (err) {
                return { success: false, error: `Cannot timeout this member. They may have Administrator permission or a role above mine. Ask the server owner to check Server Settings → Roles.` };
            }
            return { success: true, action: 'timeout', user: resolvedMember.user.tag, duration_minutes: args.duration_minutes, reason };
        }

        case 'remove_timeout': {
            if (!resolvedMember) return { success: false, error: `Could not find member "${args.user_id}" in this server.` };
            try {
                await ModerationService.removeTimeoutUser({ guild, member: resolvedMember, moderator: moderatorMember, reason });
            } catch (err) {
                return { success: false, error: `Cannot remove timeout from this member.` };
            }
            return { success: true, action: 'remove_timeout', user: resolvedMember.user.tag, reason };
        }

        case 'unban_user': {
            if (!resolvedUser) return { success: false, error: `Could not find user "${args.user_id}".` };
            try {
                await ModerationService.unbanUser({ guild, user: resolvedUser, moderator: moderatorMember, reason });
            } catch (err) {
                return { success: false, error: `Cannot unban this user.` };
            }
            return { success: true, action: 'unban', user: resolvedUser.tag, reason };
        }

        default:
            return { success: false, error: `Unknown tool: ${toolName}` };
    }
}

function splitMessage(text, maxLength = 2000) {
    const parts = [];
    while (text.length > maxLength) {
        let splitAt = text.lastIndexOf('\n', maxLength);
        if (splitAt === -1) splitAt = text.lastIndexOf(' ', maxLength);
        if (splitAt === -1) splitAt = maxLength;
        parts.push(text.slice(0, splitAt));
        text = text.slice(splitAt).trimStart();
    }
    if (text.length) parts.push(text);
    return parts;
}

async function handleAskQuestion(client, interaction, question, isModerator) {
    await InteractionHelper.safeDefer(interaction, { ephemeral: false });

    const guild = interaction.guild;
    const moderatorMember = interaction.member;

    const messages = [
        { role: 'system', content: SYSTEM_PROMPT },
    ];

    const attachment = interaction.options.getAttachment('image');
    if (attachment && attachment.contentType?.startsWith('image/')) {
        messages.push({
            role: 'user',
            content: [
                { type: 'text', text: question },
                { type: 'image_url', image_url: { url: attachment.url } },
            ],
        });
    } else {
        messages.push({ role: 'user', content: question });
    }

    const tools = (isModerator && guild) ? MODERATION_TOOLS : undefined;

    let responseText = '';
    const toolResults = [];

    try {
        let continueLoop = true;
        while (continueLoop) {
            const completion = await openai.chat.completions.create({
                model: CHAT_MODEL,
                messages,
                max_tokens: 1024,
                ...(tools ? { tools, tool_choice: 'auto' } : {}),
            });

            const choice = completion.choices?.[0];
            const assistantMsg = choice?.message;

            if (!assistantMsg) break;

            messages.push(assistantMsg);

            if (choice.finish_reason === 'tool_calls' && assistantMsg.tool_calls?.length) {
                for (const call of assistantMsg.tool_calls) {
                    let args;
                    try {
                        args = JSON.parse(call.function.arguments);
                    } catch {
                        args = {};
                    }

                    logger.info(`AI tool call: ${call.function.name}`, args);

                    let result;
                    try {
                        result = await executeTool(call.function.name, args, { guild, moderatorMember, client });
                    } catch (err) {
                        logger.error(`Tool ${call.function.name} execution error:`, err);
                        result = { success: false, error: err.message || 'Execution failed.' };
                    }

                    toolResults.push({ tool: call.function.name, ...result });

                    messages.push({
                        role: 'tool',
                        tool_call_id: call.id,
                        content: JSON.stringify(result),
                    });
                }
            } else {
                responseText = assistantMsg.content?.trim() || '';
                continueLoop = false;
            }
        }
    } catch (err) {
        logger.error('AI /ask error:', err);
        throw new TitanBotError('OpenAI API error', ErrorTypes.NETWORK, 'The AI service returned an error. Try again later.');
    }

    if (!responseText && toolResults.length === 0) {
        throw new TitanBotError('Empty AI response', ErrorTypes.UNKNOWN, 'The AI returned an empty response. Please try again.');
    }

    let reply = responseText || '';

    if (!reply && toolResults.length > 0) {
        reply = toolResults.map(r => {
            if (!r.success) return `Failed: ${r.error ?? 'Unknown error'}`;
            switch (r.action) {
                case 'ban': return `Banned **${r.user ?? r.user_id}** (deleted ${r.delete_days ?? 0}d of messages). Reason: ${r.reason}`;
                case 'kick': return `Kicked **${r.user ?? r.user_id}**. Reason: ${r.reason}`;
                case 'timeout': return `Timed out **${r.user ?? r.user_id}** for ${r.duration_minutes} min. Reason: ${r.reason}`;
                case 'remove_timeout': return `Removed timeout from **${r.user ?? r.user_id}**. Reason: ${r.reason}`;
                case 'unban': return `Unbanned **${r.user ?? r.user_id}**. Reason: ${r.reason}`;
                default: return `Action completed: ${JSON.stringify(r)}`;
            }
        }).join('\n');
    }

    if (!reply) reply = '(No response)';

    const chunks = splitMessage(reply);
    await InteractionHelper.safeEditReply(interaction, { content: chunks[0] });
    for (let i = 1; i < chunks.length; i++) {
        await interaction.followUp({ content: chunks[i] });
    }
}

async function handleGenImg(client, interaction, prompt) {
    const guildId = interaction.guild?.id;

    if (!isImageGenEnabled(guildId)) {
        throw new TitanBotError('Image gen disabled', ErrorTypes.PERMISSION, 'Image generation is currently disabled on this server.');
    }

    await InteractionHelper.safeDefer(interaction, { ephemeral: false });

    let imageData;
    try {
        const response = await openai.images.generate({
            model: IMAGE_MODEL,
            prompt,
            n: 1,
            size: '1024x1024',
            response_format: 'b64_json',
        });
        imageData = response.data?.[0]?.b64_json;
    } catch (err) {
        logger.error('AI /gen-img generation error:', err);
        throw new TitanBotError('Image generation error', ErrorTypes.NETWORK, 'Image generation failed. The model may not support this endpoint or your API key may lack access.');
    }

    if (!imageData) {
        throw new TitanBotError('No image returned', ErrorTypes.UNKNOWN, 'The AI did not return an image. Try a different prompt.');
    }

    const buffer = Buffer.from(imageData, 'base64');
    const file = new AttachmentBuilder(buffer, { name: 'generated.png' });

    const embed = createEmbed({
        title: 'Image Generated',
        description: `**Prompt:** ${prompt.slice(0, 200)}${prompt.length > 200 ? '...' : ''}`,
    })
        .setImage('attachment://generated.png')
        .setFooter({ text: `Model: ${IMAGE_MODEL}` });

    await InteractionHelper.safeEditReply(interaction, { embeds: [embed], files: [file] });
}

async function handleToggleImageGen(client, interaction, enabled) {
    if (!isBotOwner(interaction.user.id)) {
        throw new TitanBotError('Not bot owner', ErrorTypes.PERMISSION, 'Only the bot owner can use this command.');
    }

    const guildId = interaction.guild?.id;
    imageGenEnabled.set(guildId, enabled);

    const embed = createEmbed({
        title: 'Image Generation Updated',
        description: `Image generation is now **${enabled ? 'enabled' : 'disabled'}** on this server.`,
        color: enabled ? 'success' : 'warning',
    });

    await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
}

export default {
    category: 'AI',

    data: new SlashCommandBuilder()
        .setName('ai')
        .setDescription('AI-powered assistant and image generator')

        .addSubcommand((sub) =>
            sub
                .setName('ask')
                .setDescription('Ask the AI a question, or instruct it to moderate the server')
                .addStringOption((opt) =>
                    opt.setName('question').setDescription('Your question or moderation instruction').setRequired(true).setMaxLength(1000),
                )
                .addAttachmentOption((opt) =>
                    opt.setName('image').setDescription('Optional image for the AI to analyze').setRequired(false),
                ),
        )

        .addSubcommand((sub) =>
            sub
                .setName('gen-img')
                .setDescription('Generate an image from a text prompt')
                .addStringOption((opt) =>
                    opt.setName('prompt').setDescription('What to generate').setRequired(true).setMaxLength(500),
                ),
        )

        .addSubcommand((sub) =>
            sub
                .setName('toggle-images')
                .setDescription('Enable or disable image generation (bot owner only)')
                .addBooleanOption((opt) =>
                    opt.setName('enabled').setDescription('Enable or disable image generation').setRequired(true),
                ),
        ),

    async execute(interaction, config, client) {
        try {
            const sub = interaction.options.getSubcommand();

            if (sub === 'ask') {
                const question = interaction.options.getString('question');
                const isModerator = interaction.member?.permissions?.has('BanMembers') || interaction.member?.permissions?.has('KickMembers') || isBotOwner(interaction.user.id);
                await handleAskQuestion(client, interaction, question, isModerator);
            } else if (sub === 'gen-img') {
                const prompt = interaction.options.getString('prompt');
                await handleGenImg(client, interaction, prompt);
            } else if (sub === 'toggle-images') {
                const enabled = interaction.options.getBoolean('enabled');
                await handleToggleImageGen(client, interaction, enabled);
            } else {
                await InteractionHelper.safeReply(interaction, {
                    content: 'Unknown subcommand.',
                    flags: MessageFlags.Ephemeral,
                });
            }
        } catch (error) {
            await handleInteractionError(interaction, error, { command: 'ai' });
        }
    },
};
